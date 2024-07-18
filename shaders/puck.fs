#version 330

precision highp float;

in vec2 fragTexCoord;
in vec4 fragColor;

uniform vec3 LightBuffer[20];
uniform vec2 resolution;
uniform vec2 puckPos;
uniform vec2 resolution;

out vec4 finalColor;

void main()
{
    vec4 color = fragColor;
    vec2 coord = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
    vec2 localRelativeCoord = coord - puckPos;
    float localDist = length(localRelativeCoord);
    vec2 localDir = localRelativeCoord / localDist;

    for (int i = 0; i < 10; i++) {
        if (LightBuffer[i*2].x == 0 && LightBuffer[i*2].y == 0) {
            continue;
        }
        vec2 lightRelativeCoord = LightBuffer[i*2].xy - coord;
        float lightDist = length(lightRelativeCoord);
        vec2 lightDir = lightRelativeCoord / lightDist;

        float intensity = dot(localDir, lightDir);
        intensity *= pow(1 - (lightDist / length(resolution)), 2);
        intensity *= -sin(localDist * 300 / 2938.0);

        float light_intensity = LightBuffer[i*2].z;
        vec4 light_color = vec4(LightBuffer[i*2 + 1], 1.0);

        color = mix(color, light_color, intensity * 0.35 * light_intensity);
    }

    finalColor = color;
}
