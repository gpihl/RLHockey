#version 330

precision highp float;

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;
uniform vec2 resolution;
uniform vec3 PaddleBuffer[12];
uniform vec3 LightBuffer[20];
uniform vec3 ObjectBuffer[5];
uniform int paddleCount;
uniform vec2 yExtremes;

out vec4 finalColor;

const float FXAA_SPAN_MAX = 8.0;
const float FXAA_REDUCE_MUL = 1.0/8.0;
const float FXAA_REDUCE_MIN = 1.0/128.0;

float random(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

bool checkLightBlocked(vec2 lightCoord, vec2 fragCoord) {
    vec2 lightToFrag = fragCoord - lightCoord;
    float lightToFragDist = length(lightToFrag);
    vec2 lightToFragDir = lightToFrag / lightToFragDist;

    for (int i = 0; i < 5; i++) {
        if (ObjectBuffer[i].x == 0 && ObjectBuffer[i].y == 0) {
            continue;
        }
        if (ObjectBuffer[i].x == lightCoord.x && ObjectBuffer[i].y == lightCoord.y) {
            continue;
        }

        vec2 objectPos = ObjectBuffer[i].xy;
        vec2 objectToFrag = fragCoord - objectPos;
        float objectRadius = ObjectBuffer[i].z;
        if (length(objectToFrag) < objectRadius) {
            return true;
        }
        vec2 lightToObject = objectPos - lightCoord;
        float proj = dot(lightToObject, lightToFragDir);

        if (proj < 0.0 || proj > lightToFragDist) {
            continue;
        }

        vec2 closestPoint = lightCoord + proj * lightToFragDir;
        float dist = distance(closestPoint, objectPos);

        if (dist < objectRadius) {
            return true;
        }
    }

    return false;
}

vec3 getPaddleGlow(vec3 color, vec2 fragCoord) {
    for (int i = 0; i < paddleCount; i++) {
        int base = i * 3;
        vec3 paddleColor = PaddleBuffer[base];
        vec2 pos = PaddleBuffer[base+1].xy;
        float radius = PaddleBuffer[base+1].z;
        float dist = length(fragCoord - pos);
        float alpha = (dist - radius) / (radius * 0.3);
        alpha = clamp(alpha, 0.0, 2.0);
        alpha = 1.0 - cos(alpha * 3.14159265359);
        float vel_alpha = PaddleBuffer[base+2].x;
        color = mix(color, paddleColor, vel_alpha * alpha * 0.5);
    }

    return color;
}

vec3 getLightGlow(vec3 color, vec2 fragCoord) {
    for (int i = 0; i < 10; i++) {
        if (LightBuffer[i*2].x == 0 && LightBuffer[i*2].y == 0) {
            continue;
        }
        vec2 pos = LightBuffer[i*2].xy;
        if (checkLightBlocked(pos, fragCoord)) {
            continue;
        }
        float intensity = LightBuffer[i*2].z;
        float dist = length(fragCoord - pos);
        float alpha = clamp(1 - dist * 6 / length(resolution), 0.0, 1.0);
        alpha = pow(alpha, 3);

        // float dither = random(fragCoord * 0.05) * 0.02;
        // alpha += dither;

        vec3 light_color = LightBuffer[i*2 + 1];

        color = mix(color, light_color, alpha * intensity * 0.6);
    }

    return color;
}

void main()
{
    if ((1.0 - fragTexCoord.y) * resolution.y < yExtremes.x || (1.0 - fragTexCoord.y) * resolution.y > yExtremes.y) {
        finalColor = vec4(0,0,0,0);
        return;
    }

    finalColor = texture(texture0, fragTexCoord);

    if (resolution.x > 600) {

        vec2 inverseResolution = 1.0 / resolution;

        vec3 rgbNW = texture(texture0, fragTexCoord + vec2(-1.0, -1.0) * inverseResolution).xyz;
        vec3 rgbNE = texture(texture0, fragTexCoord + vec2(1.0, -1.0) * inverseResolution).xyz;
        vec3 rgbSW = texture(texture0, fragTexCoord + vec2(-1.0, 1.0) * inverseResolution).xyz;
        vec3 rgbSE = texture(texture0, fragTexCoord + vec2(1.0, 1.0) * inverseResolution).xyz;
        vec3 rgbM  = texture(texture0, fragTexCoord).xyz;

        vec3 luma = vec3(0.299, 0.587, 0.114);
        float lumaNW = dot(rgbNW, luma);
        float lumaNE = dot(rgbNE, luma);
        float lumaSW = dot(rgbSW, luma);
        float lumaSE = dot(rgbSE, luma);
        float lumaM  = dot(rgbM,  luma);

        float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
        float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

        vec2 dir;
        dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
        dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

        float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

        float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
        dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
                max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
                dir * rcpDirMin)) * inverseResolution;

        vec3 rgbA = 0.5 * (
            texture(texture0, fragTexCoord + dir * (1.0 / 3.0 - 0.5)).xyz +
            texture(texture0, fragTexCoord + dir * (2.0 / 3.0 - 0.5)).xyz);
        vec3 rgbB = rgbA * 0.5 + 0.25 * (
            texture(texture0, fragTexCoord + dir * -0.5).xyz +
            texture(texture0, fragTexCoord + dir * 0.5).xyz);

        float lumaB = dot(rgbB, luma);

        if ((lumaB < lumaMin) || (lumaB > lumaMax))
            finalColor = vec4(rgbA, 1.0);
        else
            finalColor = vec4(rgbB, 1.0);
    }

    vec2 fragCoord = vec2(fragTexCoord.x * resolution.x, (1.0 - fragTexCoord.y) * resolution.y);
    // finalColor = vec4(getPaddleGlow(finalColor.xyz, fragCoord), 1.0);
    finalColor = vec4(getLightGlow(finalColor.xyz, fragCoord), 1.0);
}
