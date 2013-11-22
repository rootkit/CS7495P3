#
# Find the KAZE includes and library
#
# This module defines
# KAZE_INCLUDE_DIR, where to find tiff.h, etc.
# KAZE_LIBRARIES, the libraries to link against to use KAZE.
# KAZE_FOUND, If false, do not try to use KAZE.

# also defined, but not for general use are
# KAZE_LIBRARY, where to find the KAZE library.
# KAZE_DEBUG_LIBRARY, where to find the KAZE library in debug
# mode.

FIND_PATH(KAZE_INCLUDE_DIR KAZE.h
  /usr/local/kaze/include
  #/usr/local/include
  #/usr/include
)

  # On unix system, debug and release have the same name
FIND_LIBRARY(KAZE_LIBRARY KAZE
	   ${KAZE_INCLUDE_DIR}/../lib
	   /usr/local/kaze
	   /usr/local/lib
	   /usr/lib)

IF(KAZE_INCLUDE_DIR)
  IF(KAZE_LIBRARY)
	SET(KAZE_FOUND "YES")
	SET(KAZE_LIBRARIES ${KAZE_LIBRARY})
  ENDIF(KAZE_LIBRARY)
ENDIF(KAZE_INCLUDE_DIR)
