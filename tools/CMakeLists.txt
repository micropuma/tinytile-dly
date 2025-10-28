# tools/CMakeLists.txt

add_llvm_executable(tutorial-compiler
    main.cpp
)

llvm_update_compile_flags(tutorial-compiler)

target_link_libraries(tutorial-compiler PRIVATE
    TutorialLib
)

# 输出路径
set_target_properties(tutorial-compiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
