#ifndef UTILITY_HLSLI
#define UTILITY_HLSLI


float UintToFloat(uint x) {
  return asfloat(0x3F800000u | (x >> 9u)) - 1.0f;
}

#endif
