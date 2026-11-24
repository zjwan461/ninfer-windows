
ninfer_add_test(ninfer_text_unicode_categories_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_unicode_categories.cpp"
  LIBRARIES ninfer_text)

ninfer_add_test(ninfer_text_llama3_split_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_llama3_split.cpp"
  LIBRARIES ninfer_text)

ninfer_add_test(ninfer_text_byte_unicode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_byte_unicode.cpp"
  LIBRARIES ninfer_text)

ninfer_add_test(ninfer_text_byte_bpe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_byte_bpe.cpp"
  LIBRARIES ninfer_text)
# The test reads its fixture and the tokenizer by relative path, so it needs to start from the source
# tree rather than from the build directory ctest would otherwise use.
set_tests_properties(ninfer_text_byte_bpe_test PROPERTIES
  SKIP_RETURN_CODE 77
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
