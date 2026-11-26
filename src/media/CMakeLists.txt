if(NINFER_WITH_MEDIA)
  add_library(ninfer_media_decode STATIC
    decode/decode.cpp
  )
  ninfer_internal_includes(ninfer_media_decode)
  target_link_libraries(ninfer_media_decode PRIVATE ${NINFER_FFMPEG_TARGET})
else()
  # Headerless build: the same symbols, refusing every call, so the numeric core links without
  # FFmpeg. Vision input is unavailable; text-only paths cannot reach the stubs.
  add_library(ninfer_media_decode STATIC
    stub.cpp
  )
  ninfer_internal_includes(ninfer_media_decode)
endif()
