ninfer_add_test(ninfer_inkling2_load_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_loading)

set_tests_properties(ninfer_inkling2_load_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_schedule_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_schedule.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_inkling2_layer0_mlp_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_layer0_mlp.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_inkling2_layer0_mlp_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_layer0_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_layer0.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_inkling2_layer0_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_nvfp4_generic_gemv_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_generic_gemv.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_inkling2_nvfp4_generic_gemv_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_layer0_artifact_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_layer0_artifact.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_inkling2_layer0_artifact_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_visits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visits.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_inkling2_visits_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_q8_generic_gemv_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_generic_gemv.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_inkling2_q8_generic_gemv_test PROPERTIES SKIP_RETURN_CODE 77)
