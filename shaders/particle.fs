#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

void main()
{
    // finalColor = vec4(fragTexCoord, 0.0, 1.0);
    // Calculate distance from the center of the particle (0.5, 0.5)
    float distance = length(fragTexCoord - vec2(0.5, 0.5));

    // Create a glow effect: the closer to the center, the more opaque the color
    float glow = 1.0 - smoothstep(0.0, 0.5, distance);
    glow = pow(glow, 8);
    // Apply the glow effect to the particle color
    finalColor = vec4(fragColor.rgb, fragColor.a * glow);
    // finalColor = vec4(fragColor.rgb * glow, 0.1);
}
