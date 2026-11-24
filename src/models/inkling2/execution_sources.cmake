target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/weight_table.cpp"
)
