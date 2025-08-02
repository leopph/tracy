#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// ----------------------------------------------------------------
// Defines
// ----------------------------------------------------------------
// - Scene can go from 0 to 2
// - The furnace_test show the energy loss, the image should be
//   all white in a perfect pathtracer
//   
// ----------------------------------------------------------------
#define SCENE 0
#define FURNACE_TEST 0
#define CAMERA_SENSITIVTY .01
#define FOCAL_LENGTH 2.5


// ---------------------------------------------
// Hash & Random
// From iq
// ---------------------------------------------
int seed = 1;

int rand(void) {
  seed = seed * 0x343fd + 0x269ec3;
  return (seed >> 16) & 32767;
}

float frand() {
  return float(rand()) / 32767.0;
}

float2 frand2() {
  return float2(frand(), frand());
}

float3 frand3() {
  return float3(frand(), frand(), frand());
}

void srand(int2 p, int frame) {
  int n = frame;
  n = (n << 13) ^ n;
  n = n * (n * n * 15731 + 789221) + 1376312589; // by Hugo Elias
  n += p.y;
  n = (n << 13) ^ n;
  n = n * (n * n * 15731 + 789221) + 1376312589;
  n += p.x;
  n = (n << 13) ^ n;
  n = n * (n * n * 15731 + 789221) + 1376312589;
  seed = n;
}

float3 hash3(float3 p) {
  uint3 x = uint3(floatBitsToUint(p));
  uint const k = 1103515245U;
  x = ((x >> 8U) ^ x.yzx) * k;
  x = ((x >> 8U) ^ x.yzx) * k;
  x = ((x >> 8U) ^ x.yzx) * k;

  return float3(x) * (1.0 / float(0xffffffffU));
}


// ---------------------------------------------
// Maths
// ---------------------------------------------
#define saturate(x) clamp(x,0.,1.)
#define PI 3.141592653589

float3x3 lookat(float3 ro, float3 ta) {
  float3 const up = float3(0., 1., 0.);
  float3 fw = normalize(ta - ro);
  float3 rt = normalize(cross(fw, normalize(up)));
  return float3x3(rt, cross(rt, fw), fw);
}

float2x2 rot(float v) {
  float a = cos(v);
  float b = sin(v);
  return float2x2(a, b, -b, a);
}

// From fizzer - https://web.archive.org/web/20170610002747/http://amietia.com/lambertnotangent.html
float3 cosineSampleHemisphere(float3 n) {
  float2 rnd = frand2();

  float a = PI * 2. * rnd.x;
  float b = 2.0 * rnd.y - 1.0;

  float3 dir = float3(sqrt(1.0 - b * b) * float2(cos(a), sin(a)), b);
  return normalize(n + dir);
}

// From pixar - https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void basis(in float3 n, out float3 b1, out float3 b2) {
  if (n.z < 0.) {
    float a = 1.0 / (1.0 - n.z);
    float b = n.x * n.y * a;
    b1 = float3(1.0 - n.x * n.x * a, -b, n.x);
    b2 = float3(b, n.y * n.y * a - 1.0, -n.y);
  } else {
    float a = 1.0 / (1.0 + n.z);
    float b = -n.x * n.y * a;
    b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
    b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
  }
}

float3 toWorld(float3 x, float3 y, float3 z, float3 v) {
  return v.x * x + v.y * y + v.z * z;
}

float3 toLocal(float3 x, float3 y, float3 z, float3 v) {
  return float3(dot(v, x), dot(v, y), dot(v, z));
}


// ---------------------------------------------
// Color
// ---------------------------------------------
float3 RGBToYCoCg(float3 rgb) {
  float y = dot(rgb, float3(1, 2, 1)) * 0.25;
  float co = dot(rgb, float3(2, 0, -2)) * 0.25 + (0.5 * 256.0 / 255.0);
  float cg = dot(rgb, float3(-1, 2, -1)) * 0.25 + (0.5 * 256.0 / 255.0);
  return float3(y, co, cg);
}

float3 YCoCgToRGB(float3 ycocg) {
  float y = ycocg.x;
  float co = ycocg.y - (0.5 * 256.0 / 255.0);
  float cg = ycocg.z - (0.5 * 256.0 / 255.0);
  return float3(y + co - cg, y + cg, y - co - cg);
}

float luma(float3 color) {
  return dot(color, float3(0.299, 0.587, 0.114));
}


// ---------------------------------------------
// Microfacet
// ---------------------------------------------
float3 F_Schlick(float3 f0, float theta) {
  return f0 + (1. - f0) * pow(1.0 - theta, 5.);
}

float F_Schlick(float f0, float f90, float theta) {
  return f0 + (f90 - f0) * pow(1.0 - theta, 5.0);
}

float D_GTR(float roughness, float NoH, float k) {
  float a2 = pow(roughness, 2.);
  return a2 / (PI * pow((NoH * NoH) * (a2 * a2 - 1.) + 1., k));
}

float SmithG(float NDotV, float alphaG) {
  float a = alphaG * alphaG;
  float b = NDotV * NDotV;
  return (2.0 * NDotV) / (NDotV + sqrt(a + b - a * b));
}

float GeometryTerm(float NoL, float NoV, float roughness) {
  float a2 = roughness * roughness;
  float G1 = SmithG(NoV, a2);
  float G2 = SmithG(NoL, a2);
  return G1 * G2;
}

float3 SampleGGXVNDF(float3 V, float ax, float ay, float r1, float r2) {
  float3 Vh = normalize(float3(ax * V.x, ay * V.y, V.z));

  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = lensq > 0. ? float3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : float3(1, 0, 0);
  float3 T2 = cross(Vh, T1);

  float r = sqrt(r1);
  float phi = 2.0 * PI * r2;
  float t1 = r * cos(phi);
  float t2 = r * sin(phi);
  float s = 0.5 * (1.0 + Vh.z);
  t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

  float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

  return normalize(float3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
}

float GGXVNDFPdf(float NoH, float NoV, float roughness) {
  float D = D_GTR(roughness, NoH, 2.);
  float G1 = SmithG(NoV, roughness * roughness);
  return (D * G1) / max(0.00001, 4.0f * NoV);
}


// ---------------------------------------------
// Sky simulation
// ---------------------------------------------
float iSphere(float3 ro, float3 rd, float radius) {
  float b = 2.0 * dot(rd, ro);
  float c = dot(ro, ro) - radius * radius;
  float disc = b * b - 4.0 * c;
  if (disc < 0.0) {
    return (-1.0);
  }
  float q = (-b + ((b < 0.0) ? -sqrt(disc) : sqrt(disc))) / 2.0;
  float t0 = q;
  float t1 = c / q;
  return max(t0, t1); //float2(t0,t1);
}

float3 skyColor(float3 rd, float3 sundir) {
#if FURNACE_TEST
    return float3(1.);
#endif
  rd.y = max(rd.y, .03);
  int const nbSamples = 16;
  int const nbSamplesLight = 16;

  float3 absR = float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
  float3 absM = float3(21e-6f);


  float3 accR = float3(0.);
  float3 accM = float3(0.);

  float mu = dot(rd, sundir);
  // mu in the paper which is the cosine of the angle between the sun direction and the ray direction 
  float g = 0.76f;
  float2 phase = float2(3.f / (16.f * PI) * (1. + mu * mu),
                        3.f / (8.f * PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(
                          1.f + g * g - 2.f * g * mu, 1.5f)));

  float radA = 6420e3;
  float radE = 6360e3;
  float3 ro = float3(0., radE + 1., 0.);
  float t = iSphere(ro, rd, radA);
  float stepSize = t / float(nbSamples);

  float2 opticalDepth = float2(0.);

  for (int i = 0; i < nbSamples; i++) {
    float3 p = ro + rd * (float(i) + .5) * stepSize;

    float h = length(p) - radE;
    float2 thickness = float2(exp(-h / 7994.), exp(-h / 1200.)) * stepSize;
    opticalDepth += thickness;

    float tl = iSphere(p, sundir, radA);
    float stepSizeLight = tl / float(nbSamplesLight);
    float2 opticalDepthLight = float2(0.);
    int j;
    for (j = 0; j < nbSamplesLight; j++) {
      float3 pl = p + sundir * (float(j) + .5) * stepSizeLight;
      float hl = length(pl) - radE;
      if (hl < 0.) {
        break;
      }
      opticalDepthLight += float2(exp(-hl / 7994.), exp(-hl / 1200.)) * stepSizeLight;
    }
    if (j == nbSamplesLight) {
      float3 tau = absR * (opticalDepth.x + opticalDepthLight.x) + absM * 1.1 * (opticalDepth.y + opticalDepthLight.y);
      float3 att = exp(-tau);
      accR += att * thickness.x;
      accM += att * thickness.y;
    }
  }

  float3 col = min((accR * absR * phase.x + accM * absM * phase.y) * 10., float3(1.));
  return col;
}


// ---------------------------------------------
// Data IO
// ---------------------------------------------
struct Data {
  float theta;
  float phi;
  float r;

  float3 ro;
  float3 ta;

  float3 oldRo;
  float3 oldTa;

  float4 oldMouse;

  float refreshTime;
};

float readData1(sampler2D tex, int id) {
  return texelFetch(tex, int2(id, 0), 0).r;
}

float3 readData3(sampler2D tex, int id) {
  return texelFetch(tex, int2(id, 0), 0).rgb;
}

float4 readData4(sampler2D tex, int id) {
  return texelFetch(tex, int2(id, 0), 0);
}

float4 writeData(float4 col, float2 fragCoord, int id, float value) {
  if (floor(fragCoord.x) == float(id)) {
    col.r = value;
  }

  return col;
}

float4 writeData(float4 col, float2 fragCoord, int id, float3 value) {
  if (floor(fragCoord.x) == float(id)) {
    col.rgb = value.rgb;
  }

  return col;
}

float4 writeData(float4 col, float2 fragCoord, int id, float4 value) {
  if (floor(fragCoord.x) == float(id)) {
    col = value;
  }

  return col;
}

Data initData() {
  Data data;

  data.theta = PI;
  data.phi = 1.;
  data.r = 9.;

  data.ro = normalize(float3(cos(data.theta), data.phi, sin(data.theta))) * data.r;
  data.ta = float3(0., 0.5, 0.);

  data.oldRo = data.ro;
  data.oldTa = data.ta;

  data.oldMouse = float4(0.);

  data.refreshTime = 0.;

  return data;
}

Data readData(sampler2D tex, float2 invRes) {
  Data data;

  data.theta = readData1(tex, 0);
  data.phi = readData1(tex, 1);
  data.r = readData1(tex, 2);

  data.ro = readData3(tex, 3);
  data.ta = readData3(tex, 4);

  data.oldRo = readData3(tex, 5);
  data.oldTa = readData3(tex, 6);

  data.oldMouse = readData4(tex, 7);
  data.refreshTime = readData1(tex, 8);
  data.ta = float3(0., 0.5, 0.);

  return data;
}

float4 writeData(float4 col, float2 fragCoord, Data data) {
  col = writeData(col, fragCoord.xy, 0, data.theta);
  col = writeData(col, fragCoord.xy, 1, data.phi);
  col = writeData(col, fragCoord.xy, 2, data.r);
  col = writeData(col, fragCoord.xy, 3, data.ro);
  col = writeData(col, fragCoord.xy, 4, data.ta);
  col = writeData(col, fragCoord.xy, 5, data.oldRo);
  col = writeData(col, fragCoord.xy, 6, data.oldTa);
  col = writeData(col, fragCoord.xy, 7, data.oldMouse);
  col = writeData(col, fragCoord.xy, 8, data.refreshTime);
  return col;
}


// ---------------------------------------------
// Distance field 
// ---------------------------------------------
float box(float3 p, float3 b) {
  float3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float map(float3 p) {
  float d = 99999.;

#if SCENE == 0
  float3 pp = p;
  p.xz = mod(p.zx, 1.) - .5;
  d = min(d, length(p - float3(0., .4, 0.)) - .4);
  //d = min(d, sheep(p*8.)/8.);
  d = max(d, abs(pp.z) - 5.);
  d = max(d, abs(pp.x) - 3.);
#endif

#if SCENE == 1
        float3 pp = p;
        p.xz = mod(p.zx,1.)-.5;
        //p.xz = rot(p.y*.5)*p.xz;
        d = min(d, box(p-float3(0.,.4,0.),float3(.4)));
        //d = min(d, sheep(p*8.)/8.);
        d = max(d, abs(pp.z)-5.);
        d = max(d, abs(pp.x)-3.);
#endif

#if SCENE == 2
    {
        float3 ip = floor(p);
        float3 fp = fract(p)-.5;

        float3 id = hash3(ip+1000.);
        fp.y = p.y-.2;
        fp.xy = rot(id.x*PI*3.) * fp.xy;
        fp.xz = rot(id.y*PI*3.) * fp.xz;
        fp.yz = rot(id.z*PI*3.) * fp.yz;
        d = min(d, box(fp,float3(.3)));
        d = max(d, abs(p.z)-5.);
        d = max(d, abs(p.x)-3.);
        
    }
#endif

#if !FURNACE_TEST
  d = min(d, p.y);
#endif
  return d;
}


// ---------------------------------------------
// Ray tracing 
// ---------------------------------------------
float trace(float3 ro, float3 rd, float2 nf) {
  float t = nf.x;
  for (int i = 0; i < 256; i++) {
    float d = map(ro + rd * t);
    if (t > nf.y || abs(d) < 0.001) {
      break;
    }
    t += d;
  }

  return t;
}

float3 normal(float3 p, float t) {
  float2 eps = float2(0.0001, 0.0);
  float d = map(p);
  float3 n;
  n.x = d - map(p - eps.xyy);
  n.y = d - map(p - eps.yxy);
  n.z = d - map(p - eps.yyx);
  n = normalize(n);
  return n;
}


#endif
