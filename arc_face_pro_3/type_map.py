from ctypes import c_char_p, c_int, c_int32, c_long, c_void_p, c_float, c_ubyte, c_byte, c_uint

# noinspection SpellCheckingInspection
MRESULT = c_long
MPChar = c_char_p  #
MPVoid = c_void_p  #
MInt32 = c_int  #
MUInt32 = c_uint
MHandle = c_void_p
MFloat = c_float  #
MByte = c_ubyte  #
MUInt8 = c_byte  #

ASF_DetectMode = c_int32
ASF_OrientPriority = c_int32
ASF_DetectModel = c_int32
ASF_RegisterOrNot = c_int32
ASF_CompareModel = c_int32
