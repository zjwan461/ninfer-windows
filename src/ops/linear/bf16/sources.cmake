target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/generic.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dual_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/gemv.cu"
)
