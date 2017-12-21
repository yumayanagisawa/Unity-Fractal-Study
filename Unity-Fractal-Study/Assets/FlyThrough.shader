// The shader code is converted from Flythrough(https://www.shadertoy.com/view/XsKXzc) created by Shane on Shadertoy

Shader "Custom/FlyThrough" {
	Properties {
		iChannel0("Albedo (RGB)", 2D) = "white" {}
		iChannel1("Albedo (RGB)", 2D) = "white" {}
	}
		SubShader{
		Tags { "RenderType" = "Opaque" }
		LOD 200
		Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag

		#include "UnityCG.cginc"

		// uniform values

		uniform sampler2D iChannel0;
		uniform sampler2D iChannel1;

		struct appdata {
			float4 vertex : POSITION;
			float2 uv : TEXCOORD0;
		};

		struct v2f {
			float2 uv : TEXCOORD0;
			float4 vertex : SV_POSITION;
		};

		v2f vert(appdata v)
		{
			v2f o;
			o.vertex = UnityObjectToClipPos(v.vertex);
			o.uv = v.uv;
			return o;
		}

		#define FAR 40.

		float hash(float n) { return frac(cos(n)*45758.5453); }

		float3 tex3D(sampler2D tex, in float3 p, in float3 n) {

			// Ryan Geiss effectively multiplies the first line by 7. It took me a while to realize
			// that it's redundant, due to the normalization that follows. I'd never noticed on account
			// of the fact that I'm not in the habit of questioning stuff written by Ryan Geiss. :)
			n = max(abs(n) - 0.2, 0.001); // n = max(abs(n), 0.001), etc.
			n /= (n.x + n.y + n.z);
			p = (tex2D(tex, p.yz)*n.x + tex2D(tex, p.zx)*n.y + tex2D(tex, p.xy)*n.z).xyz;

			// Rought sRGB to linear RGB conversion in preperation for eventual gamma correction.
			return p*p;
		}

		float map(float3 p) {

			// I'm never sure whether I should take constant stuff like the following outside the function, 
			// or not. My 1990s CPU brain tells me outside, but it doesn't seem to make a difference to frame 
			// rate in this environment one way or the other, so I'll keep it where it looks tidy. If a GPU
			// architecture\compiler expert is out there, feel free to let me know.

			static const float3 offs = float3(1, .75, .5); // Offset point.
			static const float2 a = sin(float2(0, 1.57079632) + 1.57 / 2.);
			//const mat2 m = mat2(a.y, -a.x, a);
			static const float2x2 m = float2x2(a.y, a.x, -a.x, a.y);
			static const float2 a2 = sin(float2(0, 1.57079632) + 1.57 / 4.);
			//const mat2 m2 = mat2(a2.y, -a2.x, a2);
			static const float2x2 m2 = float2x2(a2.y, a2.x, -a2.x, a2.y);

			static const float s = 5.; // Scale factor.

			float d = 1e5; // Distance.


			p = abs(frac(p*.5)*2. - 1.); // Standard spacial repetition.


			float amp = 1. / s; // Analogous to layer amplitude.


								// With only two iterations, you could unroll this for more speed,
								// but I'm leaving it this way for anyone who wants to try more
								// iterations.
			for (int i = 0; i<2; i++) {

				// Rotating.
				// for now comment
				// they should be like this
				// p.xy = vec2(p.x * m[0][0]+ p.y * m[1][0], p.x * m[0][1] + p.y * m[1][1]);
				// p.yz = vec2(p.y * m2[0][0] + p.z * m2[1][0], p.y * m2[0][1] + p.z * m2[1][1]);
				//p.xy = m*p.xy;
				//p.yz = m2*p.yz;

				p = abs(p);

				// Folding about tetrahedral planes of symmetry... I think, or is it octahedral? 
				// I should know this stuff, but topology was many years ago for me. In fact, 
				// everything was years ago. :)
				// Branchless equivalent to: if (p.x<p.y) p.xy = p.yx;
				p.xy += step(p.x, p.y)*(p.yx - p.xy);
				p.xz += step(p.x, p.z)*(p.zx - p.xz);
				p.yz += step(p.y, p.z)*(p.zy - p.yz);

				// Stretching about an offset.
				p = p*s + offs*(1. - s);

				// Branchless equivalent to:
				// if( p.z < offs.z*(1. - s)*.5)  p.z -= offs.z*(1. - s);
				p.z -= step(p.z, offs.z*(1. - s)*.5)*offs.z*(1. - s);

				// Loosely speaking, construct an object, and combine it with
				// the object from the previous iteration. The object and
				// comparison are a cube and minimum, but all kinds of 
				// combinations are possible.
				p = abs(p);
				d = min(d, max(max(p.x, p.y), p.z)*amp);

				amp /= s; // Decrease the amplitude by the scaling factor.
			}

			return d - .035; // .35 is analous to the object size.
		}

		// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total.
		float3 db(sampler2D tx, in float3 p, in float3 n, float bf) {

			const float2 e = float2(0.001, 0);
			// Gradient vector, constructed with offset greyscale texture values.
			//vec3 g = vec3( gr(tpl(tx, p - e.xyy, n)), gr(tpl(tx, p - e.yxy, n)), gr(tpl(tx, p - e.yyx, n)));

			float3x3 m = float3x3(tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));

			//float3 g = float3(0.299, 0.587, 0.114)*m;
			float3 g = float3(m[0][0] * 0.299 + m[1][0] * 0.587 + m[2][0], m[0][1] * 0.299 + m[1][1] * 0.587 + m[2][1] * 0.114, m[0][2] * 0.299 + m[1][2] * 0.587 + m[2][2] * 0.114);
			g = (g - dot(tex3D(tx, p, n), float3(0.299, 0.587, 0.114))) / e.x; g -= n*dot(n, g);

			return normalize(n + g*bf); // Bumped normal. "bf" - bump factor.

		}

		// Very basic raymarching equation.
		float trace(float3 ro, float3 rd) {


			float t = 0.;//hash(dot(rd, vec3(7, 157, 113)))*0.01;
			for (int i = 0; i< 64; i++) {

				float d = map(ro + rd*t);
				if (d < 0.0025*(1. + t*.125) || t>FAR) break;
				t += d*.75;
			}
			return t;
		}



		// Tetrahedral normal, to save a couple of "map" calls. Courtesy of IQ.
		// Apart from being faster, it can produce a subtley different aesthetic to the 6 tap version, which I sometimes prefer.
		float3 normal(in float3 p) {

			// Note the slightly increased sampling distance, to alleviate artifacts due to hit point inaccuracies.
			float2 e = float2(0.005, -0.005);
			return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx));
		}

		// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
		// Anyway, I like this one. I'm assuming it's based on IQ's original.
		float calculateAO(in float3 pos, in float3 nor)
		{
			float sca = 1.5, occ = 0.0;
			for (int i = 0; i<5; i++) {

				float hr = 0.01 + float(i)*0.5 / 4.0;
				float dd = map(nor * hr + pos);
				occ += (hr - dd)*sca;
				sca *= 0.7;
			}
			return clamp(1.0 - occ, 0.0, 1.0);
		}

		// Cheap shadows are hard. In fact, I'd almost say, shadowing repeat objects - in a setting like this - with limited 
		// iterations is impossible... However, I'd be very grateful if someone could prove me wrong. :)
		float softShadow(float3 ro, float3 lp, float k) {

			// More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
			static const int maxIterationsShad = 16;

			float3 rd = (lp - ro); // Unnormalized direction ray.

			float shade = 1.0;
			float dist = 0.05;
			float end = max(length(rd), 0.001);
			float stepDist = end / float(maxIterationsShad);

			rd /= end;

			// Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
			// number to give a decent shadow is the best one to choose. 
			for (int i = 0; i<maxIterationsShad; i++) {

				float h = map(ro + rd*dist);
				//shade = min(shade, k*h/dist);
				shade = min(shade, smoothstep(0.0, 1.0, k*h / dist)); // Subtle difference. Thanks to IQ for this tidbit.
																	  //dist += min( h, stepDist ); // So many options here: dist += clamp( h, 0.0005, 0.2 ), etc.
				dist += clamp(h, 0.02, 0.25);

				// Early exits from accumulative distance function calls tend to be a good thing.
				if (h<0.001 || dist > end) break;
			}

			// I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing.
			return min(max(shade, 0.) + 0.25, 1.0);
		}

		// Hash to return a scalar value from a 3D vector.
		float hash31(float3 p) { return frac(sin(dot(p, float3(127.1, 311.7, 74.7)))*43758.5453); }

		float drawObject(in float3 p) {

			p = frac(p) - .5;
			return dot(p, p);
		}


		float cellTile(in float3 p) {

			// Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
			float4 v, d;
			d.x = drawObject(p - float3(.81, .62, .53));
			p.xy = float2(p.y - p.x, p.y + p.x)*.7071;
			d.y = drawObject(p - float3(.39, .2, .11));
			p.yz = float2(p.z - p.y, p.z + p.y)*.7071;
			d.z = drawObject(p - float3(.62, .24, .06));
			p.xz = float2(p.z - p.x, p.z + p.x)*.7071;
			d.w = drawObject(p - float3(.2, .82, .64));

			v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y);

			d.x = min(v.z, v.w) - min(v.x, v.y); // Maximum minus second order, for that beveled Voronoi look. Range [0, 1].
												 //d.x =  min(v.x, v.y);

			return d.x*2.66; // Normalize... roughly.

		}

		float getMist(in float3 ro, in float3 rd, in float3 lp, in float t) {

			float mist = 0.;
			ro += rd*t / 64.; // Edge the ray a little forward to begin.

			for (int i = 0; i<8; i++) {
				// Lighting. Technically, a lot of these points would be
				// shadowed, but we're ignoring that.
				float sDi = length(lp - ro) / FAR;
				float sAtt = min(1. / (1. + sDi*0.25 + sDi*sDi*0.25), 1.);
				// Noise layer.
				//float n = trigNoise3D(ro/2.);//noise3D(ro/2.)*.66 + noise3D(ro/1.)*.34;
				float n = cellTile(ro / 1.);
				mist += n*sAtt;//trigNoise3D
							   // Advance the starting point towards the hit point.
				ro += rd*t / 8.;
			}

			// Add a little noise, then clamp, and we're done.
			return clamp(mist / 4. + hash31(ro)*0.2 - 0.1, 0., 1.);

		}

		float3 envMap(float3 rd, float3 n) {

			//vec3 col2 = tex3D(iChannel1, rd/4., n).zyx;//*(1.-lod*.8)
			//return smoothstep(.0, 1., col2*2.);


			// I got myself a little turned around here, but I think texture size
			// manipulation has to be performed at this level, if you want the angular
			// polar coordinates to wrap... Not sure though... It'll do. :)
			rd /= 4.;

			float2 uv = float2(atan2(rd.y, rd.x) / 6.283, acos(rd.z) / 3.14159);
			uv = frac(uv);

			float3 col = tex2D(iChannel1, uv).zyx;//*(1.-lod*.8)
			return smoothstep(.1, 1., col*col*2.);

		}


		fixed4 frag(v2f i) : SV_Target
		{
			// Unit direction ray vector: Note the absence of a divide term. I came across
			// this via a comment Shadertoy user "coyote" made. I'm pretty happy with this.
			//vec3 rd = (vec3(2.*fragCoord - iResolution.xy, iResolution.y));
			float3 rd = (float3(2.*(i.uv.xy *_ScreenParams.xy) - _ScreenParams.xy, _ScreenParams.y));

			// Barrel distortion;
			rd = normalize(float3(rd.xy, sqrt(rd.z*rd.z - dot(rd.xy, rd.xy)*0.2)));


			// Rotating the ray with Fabrice's cost cuttting matrix. I'm still pretty happy with this also. :)
			float2 m = sin(float2(1.57079632, 0) + _Time.y / 4.);
			//rd.xy = rd.xy*float2x2(m.x, m.y, -m.y, m.x);
			rd.xy = float2(m.x*rd.x + m.y*rd.y, -m.y*rd.x + m.x*rd.y);
			//rd.xz = rd.xz*float2x2(m.x, m.y, -m.y, m.x);
			rd.xz = float2(m.x*rd.x + m.y*rd.z, -m.y*rd.x + m.x*rd.z);


			// Ray origin, set off in the YZ direction. Note the "0.5." It's an old lattice trick.
			float3 ro = float3(0.0, 0.0, _Time.y * 0.1);
			float3 lp = ro + float3(0.0, .25, .25); // Light, near the ray origin.

												// Set the scene color to black.
			float3 col = float3(0, 0, 0);


			float t = trace(ro, rd); // Raymarch.

									 // Surface hit, so light it up.
			if (t<FAR) {

				float3 sp = ro + rd*t; // Surface position.
				float3 sn = normal(sp); // Surface normal.


				const float sz = 1.; // Texture size.

				sn = db(iChannel0, sp*sz, sn, .002 / (1. + t / FAR)); // Texture bump.

				float3 ref = reflect(rd, sn); // Reflected ray.


				float3 oCol = tex3D(iChannel0, sp*sz, sn); // Texture color at the surface point.
				oCol = smoothstep(0., 1., oCol);

				float sh = softShadow(sp, lp, 16.); // Soft shadows.
				float ao = calculateAO(sp, sn)*.5 + .5; // Self shadows. Not too much.

				float3 ld = lp - sp; // Light direction.
				float lDist = max(length(ld), 0.001); // Light to surface distance.
				ld /= lDist; // Normalizing the light direction vector.

				float diff = max(dot(ld, sn), 0.); // Diffuse component.
				float spec = pow(max(dot(reflect(-ld, sn), -rd), 0.), 12.); // Specular.

				float atten = 1.0 / (1.0 + lDist*0.25 + lDist*lDist*.075); // Attenuation.

																		   // Combining the elements above to light and color the scene.
				col = oCol*(diff + float3(.4, .25, .2)) + float3(1., .6, .2)*spec*2.;

				// Faux environmental mapping.
				col += envMap(reflect(rd, sn), sn);

				// Environment mapping with a cubic texture, for comparison.
				//vec3 rfCol = texture(iChannel2, reflect(rd, sn)).xyz; // Forest scene.
				//col += rfCol*rfCol*.5;


				// Shading the scene color, clamping, and we're done.
				col = min(col*atten*sh*ao, 1.);



				//col = clamp(col + hash(dot(rd, vec3(7, 157, 113)))*0.1 - 0.05, 0., 1.);

			}


			// Blend the scene and the background with some very basic, 8-layered smokey haze.
			float mist = getMist(ro, rd, lp, t);
			float3 sky = float3(.35, .6, 1)* lerp(1., .75, mist);//*(rd.y*.25 + 1.);

															// Mix the smokey haze with the object.
			col = lerp(sky, col, 1. / (t*t / FAR / FAR*128. + 1.));

			// Statistically unlikely 2.0 gamma correction, but it'll do. :)
			//fragColor = vec4(sqrt(clamp(col, 0., 1.)), 1);
			return float4(sqrt(clamp(col, 0., 1.)), 1);
		}


		ENDCG
		}

	}
	FallBack "Diffuse"
}

/*

KIFS Flythrough
---------------

After looking at Zackpudil's recent fractal shaders, I thought I'd put together something
fractal in nature. It's nothing exciting, just the standard IFS stuff you see here and there.
Like many examples, this particular one is based on Syntopia and Knighty's work.

The construction is pretty similar to that of an infinite sponge, but it has a bit of rotating,
folding, stretching, etc, thrown into the mix.

The blueish environmental lighting is experimental, and based on XT95s environment mapping in
his UI example. The idea is very simple: Instead of passing a reflective ray into a cubic
texture in cartesian form, convert it to its polar angles, then index into a 2D texture. The
results are abstract, and no substitute for the real thing, but not too bad, all things
considered.

The comments are a little rushed, but I'll tidy them up later.

Examples and references:

Menger Journey - Syntopia
https://www.shadertoy.com/view/Mdf3z7

// Explains the process in more detail.
Kaleidoscopic (escape time) IFS - Knighty
http://www.fractalforums.com/ifs-iterated-function-systems/kaleidoscopic-(escape-time-ifs)/

Ancient Generators - Zackpudil
https://www.shadertoy.com/view/4sGXzV

*/
