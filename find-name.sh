for f in /home/douliyang/local/LLVM/lib/libMLIR*.so; do 
  if nm -D -C --defined-only "$f" 2>/dev/null | grep -q "mlir::tensor::registerInferTypeOpInterfaceExternalModels"; then 
    echo "✅ Found in: $f"; 
  fi 
done
