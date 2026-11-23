ninfer_add_test(ninfer_inkling2_load_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_loading)

set_tests_properties(ninfer_inkling2_load_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_schedule_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_schedule.cpp"
  LIBRARIES ninfer_model_loading)
