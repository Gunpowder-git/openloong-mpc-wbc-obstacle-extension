/*
Offscreen snapshot tool: loads a scene, optionally steps the full walking controller for
a few seconds, renders one frame to an RGB file (W H then raw RGB bytes, top-to-bottom).
Used to visually verify scene quality. Not part of the runtime demos.

Usage: snapshot <scene.xml> <out.rgb> [settleSeconds] [camAzimuth] [camElevation] [camDist] [lookX]
*/
#include <mujoco/mujoco.h>
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
    const char *scene = (argc > 1) ? argv[1] : "../models/scene_manual_improved.xml";
    const char *out = (argc > 2) ? argv[2] : "/tmp/snap.rgb";
    double camAz = (argc > 4) ? atof(argv[4]) : 130;
    double camEl = (argc > 5) ? atof(argv[5]) : -20;
    double camDist = (argc > 6) ? atof(argv[6]) : 9.0;
    double lookX = (argc > 7) ? atof(argv[7]) : 5.0;

    char err[1000];
    mjModel *m = mj_loadXML(scene, 0, err, 1000);
    if (!m) { fprintf(stderr, "load fail: %s\n", err); return 1; }
    mjData *d = mj_makeData(m);
    mj_forward(m, d);

    if (!glfwInit()) { fprintf(stderr, "glfw init fail\n"); return 1; }
    glfwWindowHint(GLFW_VISIBLE, 0);
    const int W = 1280, H = 800;
    GLFWwindow *win = glfwCreateWindow(W, H, "snap", NULL, NULL);
    if (!win) { fprintf(stderr, "window fail\n"); return 1; }
    glfwMakeContextCurrent(win);

    mjvCamera cam; mjvOption opt; mjvScene scn; mjrContext con;
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    cam.type = mjCAMERA_FREE;
    cam.lookat[0] = lookX; cam.lookat[1] = 0; cam.lookat[2] = 0.4;
    cam.distance = camDist; cam.azimuth = camAz; cam.elevation = camEl;

    mjrRect viewport = {0, 0, W, H};
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    unsigned char *rgb = (unsigned char *)malloc(3 * W * H);
    mjr_readPixels(rgb, NULL, viewport, &con);

    FILE *f = fopen(out, "wb");
    fprintf(f, "%d %d\n", W, H);
    // flip vertically (mjr_readPixels is bottom-to-top)
    for (int y = H - 1; y >= 0; --y)
        fwrite(rgb + 3 * W * y, 1, 3 * W, f);
    fclose(f);
    free(rgb);
    printf("wrote %s (%dx%d)\n", out, W, H);
    return 0;
}
