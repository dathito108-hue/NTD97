package ai.ntd97.mobile;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.view.MotionEvent;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public final class NtdAvatarGLSurfaceView extends GLSurfaceView {
    private final AvatarRenderer renderer;
    private float lastX;
    private float lastY;

    public NtdAvatarGLSurfaceView(Context context) {
        super(context);
        setEGLContextClientVersion(2);
        renderer = new AvatarRenderer();
        setRenderer(renderer);
        setRenderMode(RENDERMODE_CONTINUOUSLY);
    }

    public void setAvatarState(NtdRuntimeHost.AvatarState state) {
        renderer.setState(state == null ? NtdRuntimeHost.AvatarState.idle() : state);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN) {
            lastX = event.getX();
            lastY = event.getY();
            return true;
        }

        if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
            float dx = event.getX() - lastX;
            float dy = event.getY() - lastY;
            renderer.addUserRotation(dx * 0.2f, dy * 0.2f);
            lastX = event.getX();
            lastY = event.getY();
            return true;
        }
        return true;
    }

    private static final class AvatarRenderer implements Renderer {
        private static final String VERTEX_SHADER =
                "uniform mat4 uMvp; attribute vec4 aPosition;" +
                "void main(){ gl_Position = uMvp * aPosition; }";
        private static final String FRAGMENT_SHADER =
                "precision mediump float; uniform vec4 uColor;" +
                "void main(){ gl_FragColor = uColor; }";

        private static final float[] VERTICES = {
                0.0f, 1.0f, 0.0f,  -0.8f, -0.4f, 0.8f,   0.8f, -0.4f, 0.8f,
                0.0f, 1.0f, 0.0f,   0.8f, -0.4f, 0.8f,   0.8f, -0.4f, -0.8f,
                0.0f, 1.0f, 0.0f,   0.8f, -0.4f, -0.8f, -0.8f, -0.4f, -0.8f,
                0.0f, 1.0f, 0.0f,  -0.8f, -0.4f, -0.8f, -0.8f, -0.4f, 0.8f,
                -0.8f, -0.4f, 0.8f, -0.8f, -0.4f, -0.8f, 0.8f, -0.4f, -0.8f,
                -0.8f, -0.4f, 0.8f, 0.8f, -0.4f, -0.8f, 0.8f, -0.4f, 0.8f
        };

        private final FloatBuffer vertexBuffer;
        private final float[] projection = new float[16];
        private final float[] view = new float[16];
        private final float[] model = new float[16];
        private final float[] tmp = new float[16];
        private final float[] mvp = new float[16];

        private volatile NtdRuntimeHost.AvatarState state = NtdRuntimeHost.AvatarState.idle();
        private volatile float userYaw;
        private volatile float userPitch;

        private int program;
        private int positionLocation;
        private int mvpLocation;
        private int colorLocation;

        AvatarRenderer() {
            vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
            vertexBuffer.put(VERTICES).position(0);
        }

        void setState(NtdRuntimeHost.AvatarState state) {
            this.state = state;
        }

        void addUserRotation(float yaw, float pitch) {
            userYaw = clamp(userYaw + yaw, -45.0f, 45.0f);
            userPitch = clamp(userPitch + pitch, -25.0f, 25.0f);
        }

        @Override
        public void onSurfaceCreated(
                javax.microedition.khronos.egl.EGLConfig config) {
            GLES20.glClearColor(0.03f, 0.03f, 0.05f, 0.0f);
            GLES20.glEnable(GLES20.GL_DEPTH_TEST);
            program = buildProgram(VERTEX_SHADER, FRAGMENT_SHADER);
            positionLocation = GLES20.glGetAttribLocation(program, "aPosition");
            mvpLocation = GLES20.glGetUniformLocation(program, "uMvp");
            colorLocation = GLES20.glGetUniformLocation(program, "uColor");
        }

        @Override
        public void onSurfaceChanged(
                javax.microedition.khronos.opengles.GL10 gl,
                int width,
                int height) {
            GLES20.glViewport(0, 0, width, height);
            float ratio = width / (float) Math.max(1, height);
            Matrix.perspectiveM(projection, 0, 50.0f, ratio, 0.1f, 100.0f);
            Matrix.setLookAtM(view, 0, 0.0f, 0.2f, 4.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
        }

        @Override
        public void onDrawFrame(javax.microedition.khronos.opengles.GL10 gl) {
            NtdRuntimeHost.AvatarState snapshot = state;
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

            Matrix.setIdentityM(model, 0);
            float gazeYaw = snapshot.gazeX / 1000.0f * 18.0f;
            float gazePitch = -snapshot.gazeY / 1000.0f * 10.0f;
            Matrix.rotateM(model, 0, userYaw + gazeYaw, 0.0f, 1.0f, 0.0f);
            Matrix.rotateM(model, 0, userPitch + gazePitch, 1.0f, 0.0f, 0.0f);

            float speakingScale = snapshot.speaking
                    ? 1.0f + (snapshot.lipAmplitude / 1000.0f) * 0.04f
                    : 1.0f;
            Matrix.scaleM(model, 0, 1.0f, speakingScale, 1.0f);

            Matrix.multiplyMM(tmp, 0, view, 0, model, 0);
            Matrix.multiplyMM(mvp, 0, projection, 0, tmp, 0);

            float[] color = colorFor(snapshot.expression, snapshot.mode);
            GLES20.glUseProgram(program);
            GLES20.glEnableVertexAttribArray(positionLocation);
            GLES20.glVertexAttribPointer(
                    positionLocation,
                    3,
                    GLES20.GL_FLOAT,
                    false,
                    3 * 4,
                    vertexBuffer);
            GLES20.glUniformMatrix4fv(mvpLocation, 1, false, mvp, 0);
            GLES20.glUniform4fv(colorLocation, 1, color, 0);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES.length / 3);
            GLES20.glDisableVertexAttribArray(positionLocation);
        }

        private static float[] colorFor(int expression, int mode) {
            if (mode == 9) {
                return new float[]{0.85f, 0.25f, 0.25f, 1.0f};
            }
            if (mode == 5) {
                return new float[]{0.95f, 0.65f, 0.20f, 1.0f};
            }
            if (expression == 6 || mode == 8) {
                return new float[]{0.20f, 0.85f, 0.55f, 1.0f};
            }
            if (expression == 2 || mode == 3 || mode == 4) {
                return new float[]{0.25f, 0.55f, 0.95f, 1.0f};
            }
            return new float[]{0.55f, 0.65f, 0.85f, 1.0f};
        }

        private static int buildProgram(String vertexSource, String fragmentSource) {
            int vertex = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource);
            int fragment = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
            int program = GLES20.glCreateProgram();
            GLES20.glAttachShader(program, vertex);
            GLES20.glAttachShader(program, fragment);
            GLES20.glLinkProgram(program);
            return program;
        }

        private static int compileShader(int type, String source) {
            int shader = GLES20.glCreateShader(type);
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            return shader;
        }

        private static float clamp(float value, float min, float max) {
            return Math.max(min, Math.min(max, value));
        }
    }
}
