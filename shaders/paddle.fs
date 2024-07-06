#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

uniform vec3 LightBuffer[4];
uniform vec2 resolution;
uniform vec2 paddlePos;

out vec4 finalColor;

void main()
{
    // vec4 color = vec4(0,0,0,1);
    vec4 color = fragColor;
    for (int i = 0; i < 4; i++) {
        vec2 coord = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
        vec2 localRelativeCoord = coord - paddlePos;
        float localDist = length(localRelativeCoord);
        vec2 localDir = localRelativeCoord / localDist;

        vec2 lightRelativeCoord = LightBuffer[i].xy - coord;
        float lightDist = length(lightRelativeCoord);
        vec2 lightDir = lightRelativeCoord / lightDist;

        float intensity = dot(localDir, lightDir);
        intensity *= pow(1 - (lightDist / length(resolution)), 2);

        intensity *= pow(localDist * 100 / length(resolution), 2);

        float light_intensity = LightBuffer[i].z;

        color = mix(color, vec4(1,1,1,1), intensity * 0.3 * light_intensity);
    }

    finalColor = color;

    // vec4 color = fragColor;
    // for (int i = 0; i < 4; i++) {
    //     vec2 lightDir = LightDir[i];
    //     float specularAlpha = dot(relativeCoord, lightDir);
    //     color = mix(color, vec4(1,1,1,1), specularAlpha);
    // }

    // vec2 lightDir = LightDir[0];
    // float specularAlpha = dot(relativeCoord, lightDir) + 0.5;
    // specularAlpha = clamp(specularAlpha, 0, 1);
    // color = mix(color, vec4(1,1,1,1), specularAlpha);

    // finalColor = color;

    // vec2 lightPos = LightBuffer[0].xy;
    // vec2 coord = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
    // float intensity = length(lightPos - coord) / length(resolution);
    // finalColor = vec4(intensity, intensity, intensity, 1);
    // finalColor = vec4(coord / length(resolution), 0, 1);
}
