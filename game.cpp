#include "raylib.h"
#include <cmath>
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {

    const int screenWidth = 1000;
    const int screenHeight = 600;

    // -------------------------
    // ONNX RUNTIME SETUP
    // -------------------------

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "gesture-football"
    );

    Ort::SessionOptions sessionOptions;

    sessionOptions.SetIntraOpNumThreads(1);

    Ort::Session session(
        env,
        "gesture_lstm.onnx",
        sessionOptions
    );

    std::cout
        << "ONNX model loaded successfully."
        << std::endl;


    // -------------------------
    // GAME SETUP
    // -------------------------

    InitWindow(
        screenWidth,
        screenHeight,
        "Gesture Football"
    );

    SetTargetFPS(60);


    Vector2 player = {
        500.0f,
        450.0f
    };

    Vector2 ball = {
        500.0f,
        410.0f
    };

    Vector2 ballVelocity = {
        0.0f,
        0.0f
    };

    const float playerSpeed = 5.0f;

    float playerAngle = -90.0f;

    bool holdingBall = true;


    while (!WindowShouldClose()) {

        // -------------------------
        // PLAYER MOVEMENT
        // -------------------------

        Vector2 movement = {
            0.0f,
            0.0f
        };


        if (IsKeyDown(KEY_A)) {
            movement.x -= 1.0f;
        }

        if (IsKeyDown(KEY_D)) {
            movement.x += 1.0f;
        }

        if (IsKeyDown(KEY_W)) {
            movement.y -= 1.0f;
        }

        if (IsKeyDown(KEY_S)) {
            movement.y += 1.0f;
        }


        if (
            movement.x != 0.0f ||
            movement.y != 0.0f
        ) {

            float length = sqrt(
                movement.x * movement.x +
                movement.y * movement.y
            );

            movement.x /= length;
            movement.y /= length;


            player.x += movement.x * playerSpeed;
            player.y += movement.y * playerSpeed;


            playerAngle = atan2(
                movement.y,
                movement.x
            ) * RAD2DEG;
        }


        // -------------------------
        // FIELD BOUNDARIES
        // -------------------------

        if (player.x < 75) {
            player.x = 75;
        }

        if (player.x > 925) {
            player.x = 925;
        }

        if (player.y < 75) {
            player.y = 75;
        }

        if (player.y > 525) {
            player.y = 525;
        }


        // -------------------------
        // ACTIONS
        // -------------------------

        // PASS LEFT
        if (IsKeyPressed(KEY_J)) {

            holdingBall = false;

            ballVelocity = {
                -7.0f,
                -2.0f
            };
        }


        // PASS RIGHT
        if (IsKeyPressed(KEY_L)) {

            holdingBall = false;

            ballVelocity = {
                7.0f,
                -2.0f
            };
        }


        // SHOOT
        if (IsKeyPressed(KEY_SPACE)) {

            holdingBall = false;

            ballVelocity = {
                cos(playerAngle * DEG2RAD) * 12.0f,
                sin(playerAngle * DEG2RAD) * 12.0f
            };
        }


        // THROUGH BALL
        if (IsKeyPressed(KEY_I)) {

            holdingBall = false;

            ballVelocity = {
                cos(playerAngle * DEG2RAD) * 7.0f,
                sin(playerAngle * DEG2RAD) * 7.0f
            };
        }


        // HOLD
        if (IsKeyPressed(KEY_K)) {

            holdingBall = true;

            ballVelocity = {
                0.0f,
                0.0f
            };
        }


        // -------------------------
        // BALL UPDATE
        // -------------------------

        if (holdingBall) {

            ball.x =
                player.x +
                cos(playerAngle * DEG2RAD) * 40.0f;

            ball.y =
                player.y +
                sin(playerAngle * DEG2RAD) * 40.0f;

        } else {

            ball.x += ballVelocity.x;
            ball.y += ballVelocity.y;

            ballVelocity.x *= 0.98f;
            ballVelocity.y *= 0.98f;
        }


        // -------------------------
        // RESET BALL
        // -------------------------

        if (
            ball.x < 50 ||
            ball.x > 950 ||
            ball.y < 50 ||
            ball.y > 550
        ) {

            holdingBall = true;

            ballVelocity = {
                0.0f,
                0.0f
            };
        }


        // -------------------------
        // PLAYER DIRECTION
        // -------------------------

        Vector2 front = {

            player.x +
            cos(playerAngle * DEG2RAD) * 35.0f,

            player.y +
            sin(playerAngle * DEG2RAD) * 35.0f
        };


        // -------------------------
        // DRAW
        // -------------------------

        BeginDrawing();

        ClearBackground(DARKGREEN);


        DrawRectangleLines(
            50,
            50,
            900,
            500,
            WHITE
        );


        DrawLine(
            500,
            50,
            500,
            550,
            WHITE
        );


        DrawCircleLines(
            500,
            300,
            70,
            WHITE
        );


        DrawCircleV(
            player,
            25,
            BLUE
        );


        DrawLineEx(
            player,
            front,
            6.0f,
            YELLOW
        );


        DrawCircleV(
            ball,
            12,
            WHITE
        );


        DrawText(
            "WASD: Move",
            20,
            10,
            20,
            WHITE
        );

        DrawText(
            "J: Pass Left",
            20,
            35,
            20,
            WHITE
        );

        DrawText(
            "L: Pass Right",
            20,
            60,
            20,
            WHITE
        );

        DrawText(
            "SPACE: Shoot",
            20,
            85,
            20,
            WHITE
        );

        DrawText(
            "I: Through Ball",
            20,
            110,
            20,
            WHITE
        );

        DrawText(
            "K: Hold",
            20,
            135,
            20,
            WHITE
        );


        EndDrawing();
    }


    CloseWindow();

    return 0;
}