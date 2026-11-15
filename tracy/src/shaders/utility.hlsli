#ifndef UTILITY_HLSLI
#define UTILITY_HLSLI


static float const kPi = 3.14159265359f;


float UintToFloat(uint x) {
  return asfloat(0x3F800000u | (x >> 9u)) - 1.0f;
}

#endif
