#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

void main()
{
    float distance = length(fragTexCoord - vec2(0.5, 0.5));
    float glow = 1.0 - smoothstep(0.0, 0.5, distance);
    glow = pow(glow, 8);
    finalColor = vec4(fragColor.rgb, fragColor.a * glow);
}
