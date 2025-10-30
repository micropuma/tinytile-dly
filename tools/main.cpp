#include "Passes.h"
#include "Tutorial.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Debug/DebuggerExecutionContextHook.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace llvm;

const char *toolName = "Tensor Tiling Tutorial Compiler";

void createPassPipeline(PassManager &pm) {
  // Apply required transform spec.
  { pm.addPass(tutorial::createTutorialApplyTilingSpec()); }

  // Parallel tiling using scf.forall
  // TODO(leon): deep dive here
  {
    tutorial::TutorialTileAndFuseOptions options;
    options.tilingLevel = tutorial::TilingLevel::Parallel;
    pm.addPass(tutorial::createTutorialTileAndFuse(options));
  }

  // Reduction tiling using scf.for
  {
    tutorial::TutorialTileAndFuseOptions options;
    options.tilingLevel = tutorial::TilingLevel::Reduction;
    pm.addPass(tutorial::createTutorialTileAndFuse(options));
  }

  // Generalization and cleanups.
  {
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createLoopInvariantCodeMotionPass());
    LinalgFoldUnitExtentDimsPassOptions options;
    options.useRankReducingSlices = true;
    pm.addPass(createLinalgFoldUnitExtentDimsPass(options));
  }

  // Vectorization
  // TODO(leon): deep dive here
  {
    pm.addPass(tutorial::createTutorialVectorization());

    // followed by cleanups
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(tensor::createFoldTensorSubsetOpsPass());
    pm.addPass(createLoopInvariantSubsetHoistingPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Bufferization
  // TODO(leon): deep dive here
  {
    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    // options.functionBoundaryTypeConversion = bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
  }
}

LogicalResult tutorialOpt(int argc, char **argv) {
  static llvm::cl::OptionCategory mainOptions("Tutorial Options");

  // General command line flags.
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename(
      "o", cl::desc("Output filename"), cl::value_desc("filename"),
      cl::init("-"), llvm::cl::cat(mainOptions));

  cl::ParseCommandLineOptions(argc, argv);

  InitLLVM y(argc, argv);

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());

  DialectRegistry registry;

  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<index::IndexDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<transform::TransformDialect>();
  registry.insert<tutorial::TutorialDialect>();

  linalg::registerAllDialectInterfaceImplementations(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  affine::registerValueBoundsOpInterfaceExternalModels(registry);

  tensor::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);

  // Create a context just for the current buffer. Disable threading on creation
  // since we'll inject the thread-pool separately.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);

  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);

  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> op =
      parseSourceFileForTool(sourceMgr, parseConfig, true);
  if (!op) {
    return failure();
  }

  PassManager pm(op.get()->getName(), PassManager::Nesting::Implicit);
  pm.enableVerifier();
  pm.enableIRPrinting();

  createPassPipeline(pm);

  // Run the pipeline.
  if (failed(pm.run(*op))) {
    return failure();
  }

  AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr);
  op.get()->print(output->os(), asmState);
  output->os() << '\n';

  output->keep();
  return success();
}

int main(int argc, char **argv) {
  return mlir::asMainReturnCode(tutorialOpt(argc, argv));
}
