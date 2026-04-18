using UnityEngine;
using System.Net.Sockets;
using System;
using System.Diagnostics;
using Debug = UnityEngine.Debug;

public class SwarmClient : MonoBehaviour
{
    [Header("Overfit Engine: Connection")]
    public string ServerHost = "127.0.0.1";
    public int ServerPort = 5000;

    [Header("Swarm Settings")]
    public int SwarmSize = 100_000;
    public Transform Target;
    public Transform Predator;

    [Header("Target AI (auto teleport during training)")]
    public bool AutoTeleportTarget = true;
    public int TargetTeleportInterval = 300;  // frames
    public float TargetRange = 18f;

    [Header("Rendering")]
    public Mesh BotMesh;
    public Material BotMaterial;
    public float BotScale = 0.4f;

    // Network buffers
    private byte[] _sendBuf;
    private byte[] _recvBuf;
    private Matrix4x4[] _matrices;

    private TcpClient _client;
    private NetworkStream _stream;
    private long _lastFrameMs;
    private int _frameCount;

    void Start()
    {
        // 16 bytes: target(x,z), predator(x,z)
        _sendBuf = new byte[16];

        // Per-bot: 2 floats (posX, posZ) = 8 bytes
        _recvBuf = new byte[SwarmSize * 8];

        _matrices = new Matrix4x4[SwarmSize];

        // Initialize with zeros — first frame positions come from server
        for (int i = 0; i < SwarmSize; i++)
            _matrices[i] = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Vector3.one * BotScale);

        try
        {
            _client = new TcpClient(ServerHost, ServerPort);
            _client.NoDelay = true;
            _stream = _client.GetStream();
            Debug.Log($"<color=cyan>CONNECTED:</color> Overfit Server at {ServerHost}:{ServerPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"CONNECTION FAILED. Is server running? Err: {e.Message}");
        }
    }

    void Update()
    {
        if (_stream == null)
        {
            return;
        }

        // Auto-teleport target for varied training
        if (AutoTeleportTarget && _frameCount % TargetTeleportInterval == 0 && _frameCount > 0)
        {
            Target.position = new Vector3(
                UnityEngine.Random.Range(-TargetRange, TargetRange),
                0.5f,
                UnityEngine.Random.Range(-TargetRange, TargetRange));
        }
        _frameCount++;

        var sw = Stopwatch.StartNew();

        try
        {
            // 1. Pack command: [targetX, targetZ, predatorX, predatorZ]
            var tPos = Target.position;
            var pPos = Predator.position;

            // Explicit little-endian write (server uses BinaryPrimitives.ReadSingleLittleEndian)
            WriteFloatLE(_sendBuf, 0, tPos.x);
            WriteFloatLE(_sendBuf, 4, tPos.z);
            WriteFloatLE(_sendBuf, 8, pPos.x);
            WriteFloatLE(_sendBuf, 12, pPos.z);

            _stream.Write(_sendBuf, 0, _sendBuf.Length);

            // 2. Receive all bot positions
            int totalRead = 0;
            while (totalRead < _recvBuf.Length)
            {
                int r = _stream.Read(_recvBuf, totalRead, _recvBuf.Length - totalRead);
                if (r == 0)
                {
                    Debug.LogWarning("Server closed connection.");
                    _stream = null;
                    return;
                }
                totalRead += r;
            }

            // 3. Build render matrices directly from received positions
            // (Server sends XZ pairs, we map to world XYZ with Y fixed)
            BuildMatricesFromBuffer();

            // 4. GPU Instanced render
            var rp = new RenderParams(BotMaterial)
            {
                worldBounds = new Bounds(Vector3.zero, Vector3.one * 200f)
            };
            Graphics.RenderMeshInstanced(rp, BotMesh, 0, _matrices);
        }
        catch (Exception e)
        {
            Debug.LogError($"Network error: {e.Message}");
            _stream = null;
        }

        _lastFrameMs = sw.ElapsedMilliseconds;
    }

    private unsafe void BuildMatricesFromBuffer()
    {
        // Treat _recvBuf as float[] via unsafe pointer — zero-copy
        fixed (byte* pBytes = _recvBuf)
        {
            float* pFloats = (float*)pBytes;
            for (int i = 0; i < SwarmSize; i++)
            {
                float px = pFloats[i * 2 + 0];
                float pz = pFloats[i * 2 + 1];
                _matrices[i] = Matrix4x4.TRS(
                    new Vector3(px, 0.5f, pz),
                    Quaternion.identity,
                    Vector3.one * BotScale);
            }
        }
    }

    private static void WriteFloatLE(byte[] buf, int offset, float value)
    {
        // BitConverter.GetBytes is always little-endian on x86/ARM
        byte[] bytes = BitConverter.GetBytes(value);
        // Copy in-place (BitConverter may allocate; caller should prefer stackalloc if hot)
        Buffer.BlockCopy(bytes, 0, buf, offset, 4);
    }

    void OnGUI()
    {
        var style = new GUIStyle
        {
            fontSize = 22,
            fontStyle = FontStyle.Bold,
            normal = { textColor = Color.cyan }
        };

        string stats =
            $"OVERFIT SWARM\n" +
            $"Population: {SwarmSize:N0}\n" +
            $"Frame time: {_lastFrameMs}ms\n" +
            $"FPS: {Mathf.Round(1f / Time.smoothDeltaTime)}\n" +
            $"Target: ({Target.position.x:F1}, {Target.position.z:F1})\n" +
            $"Predator: ({Predator.position.x:F1}, {Predator.position.z:F1})";

        GUI.Label(new Rect(20, 20, 600, 300), stats, style);
    }

    void OnApplicationQuit()
    {
        _stream?.Close();
        _client?.Close();
    }
}