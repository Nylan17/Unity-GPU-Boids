using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Runtime.InteropServices;

using System.IO; // starting my stuff
using MathNet.Numerics.LinearRegression;
using MathNet.Numerics.LinearAlgebra;



public class MetricsRecorder
{
    private List<float> avgVelocities = new List<float>();
    private List<Vector3> avgPositions = new List<Vector3>();
    private List<float> localOrders = new List<float>();
    private List<float> temperatures = new List<float>();

    public void AddMetrics(float avgVelocity, Vector3 avgPosition, float avgLocalOrder, float temperature)
    {
        avgVelocities.Add(avgVelocity);
        avgPositions.Add(avgPosition);
        localOrders.Add(avgLocalOrder);
        temperatures.Add(temperature);
    }

    public void SaveMetrics(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("AvgVelocity,AvgPositionX,AvgPositionY,AvgPositionZ,LocalOrder,Temperature");

            for (int i = 0; i < avgVelocities.Count; i++)
            {
                string line = avgVelocities[i].ToString("F6") + "," +
                              avgPositions[i].x.ToString("F6") + "," +
                              avgPositions[i].y.ToString("F6") + "," +
                              avgPositions[i].z.ToString("F6") + "," +
                              localOrders[i].ToString("F6") + "," +
                              temperatures[i].ToString("F6");

                writer.WriteLine(line);
            }
        }
    }

    public float[][] GetMetrics()
    {
        int count = avgVelocities.Count;
        float[][] metrics = new float[count][];

        for (int i = 0; i < count; i++)
        {
            metrics[i] = new float[]
            {
                avgVelocities[i],
                avgPositions[i].x,
                avgPositions[i].y,
                avgPositions[i].z,
                localOrders[i]
            };
        }

        return metrics;
    }

    public float[] GetTemperatures()
    {
        return temperatures.ToArray();
    }

    public void Clear()
    {
        avgVelocities.Clear();
        avgPositions.Clear();
        localOrders.Clear();
        temperatures.Clear();
    }
}


public class DataSplitter
{
    public static void SplitData(float[] inputData, out float[] trainData, out float[] testData, float trainRatio = 0.7f)
    {
        int trainSize = (int)(inputData.Length * trainRatio);
        trainData = new float[trainSize];
        testData = new float[inputData.Length - trainSize];

        for (int i = 0; i < trainSize; i++)
        {
            trainData[i] = inputData[i];
        }

        for (int i = trainSize; i < inputData.Length; i++)
        {
            testData[i - trainSize] = inputData[i];
        }
    }
}


public class RidgeRegression
{
    private Vector<float> weights;
    private float lambda;

    public RidgeRegression(float lambda = 0.1f)
    {
        this.lambda = lambda;
    }

    public float[] GetWeights()
    {
        return weights.ToArray();
    }

    public void Train(float[][] X, float[] y)
    {
        var matrixX = Matrix<float>.Build.DenseOfRowArrays(X);
        var vectorY = Vector<float>.Build.Dense(y);

        var identity = Matrix<float>.Build.DenseIdentity(matrixX.ColumnCount);
        var regularization = identity.Multiply(lambda);

        var XtX = matrixX.TransposeThisAndMultiply(matrixX);
        var regularizedXtX = XtX.Add(regularization);

        var Xty = matrixX.TransposeThisAndMultiply(vectorY);

        weights = regularizedXtX.Solve(Xty);
    }

    public float Predict(float[] features)
    {
        var vectorX = Vector<float>.Build.Dense(features);
        return vectorX.DotProduct(weights);
    }
}


public class GPUFlock : MonoBehaviour {
    public struct GPUBoid
    {
        public Vector3 position;
        public Vector3 direction;
        public float noise_offset;
        public float speed;
        public float frame;
        public float next_frame;
        public float frame_interpolation;
        public float size;
    }

    public struct GPUAffector {
        public Vector3 position;
        public float force;
        public float distance;
        public int axis;
        public Vector2 padding;
    }

    public ComputeShader _ComputeFlock;
    public GameObject TargetBoidToGPUSkin;
    public Transform Target;
    public Mesh BoidMesh;
    public Material BoidMaterial;

    private SkinnedMeshRenderer BoidSMR;
    private Animator _Animator;
    public AnimationClip _AnimationClip;
    private int NbFramesInAnimation;

    public bool UseAffectors;
    public TextAsset DrawingAffectors;
    public bool UseMeshAffectors = false;
    public Mesh MeshAffectors;    
    public float ScaleDrawingAffectors = 0.03f;
    public bool ReverseYAxisDrawingAffectors = true;
    public Vector3 DrawingAffectorsOffset;
    public bool DrawDrawingAffectors = true;
    private int NbAffectors = 0;

    public int BoidsCount;
    public int StepBoidCheckNeighbours = 1;
    public float SpawnRadius;
    public float RotationSpeed = 4f;
    public float BoidSpeed = 6f;
    public float NeighbourDistance = 2f;
    public float BoidSpeedVariation = 0.9f;
    public float BoidFrameSpeed = 10f;
    public bool FrameInterpolation = true;
    public float AffectorForce = 2f;
    public float AffectorDistance = 2f;
    public float MaxAffectorFullAxisSize = 20f;
    private GPUBoid[] boidsData;
    private GPUAffector[] Affectors = new GPUAffector[1];

    private int kernelHandle;
    private ComputeBuffer BoidBuffer;
    private ComputeBuffer AffectorBuffer;
    private ComputeBuffer VertexAnimationBuffer;
    private ComputeBuffer _drawArgsBuffer;
    private Bounds InfiniteBounds = new Bounds(Vector3.zero, Vector3.one * 9999);

    private const int THREAD_GROUP_SIZE = 256;

    public bool UseTemperature = true;
    public float Temperature = 1.0f; // Default temperature
    public TextAsset TemperatureDataFile;
    private float[] temperatureData;
    private int currentTemperatureIndex = 0;

    private ComputeBuffer avgVelocityBuffer;
    private ComputeBuffer avgPositionBuffer;
    private ComputeBuffer localOrderBuffer;

    private float[] featureMeans;
    private float[] featureStdDevs;

    public int warmupPeriod = 500;
    private int currentFrame = 0;
    private bool isWarmedUp = false;

    private ComputeBuffer outputSumPositionBuffer;
    private ComputeBuffer outputSumVelocityBuffer;
    private ComputeBuffer outputSumOrderBuffer;
    private ComputeBuffer outputTotalBoidsBuffer;

    private uint[] sumPositionData = new uint[3];
    private uint[] sumVelocityData = new uint[1];
    private uint[] sumOrderData = new uint[1];
    private uint[] totalBoidsData = new uint[1];

    private const float SCALE_FACTOR = 1000000f;

    private float[] avgVelocityData;
    private Vector3[] avgPositionData;
    private float[] localOrderData;
    private float[] trainData, testData;
    private List<float> predictions = new List<float>();

    [SerializeField]
    private float temperatureInputScale = 1.0f;


    private MetricsRecorder metricsRecorder = new MetricsRecorder();

    void StoreMetrics(float avgVelocity, Vector3 avgPosition, float avgLocalOrder)
    {
        // Log raw values before any processing
        //Debug.Log($"Raw Metrics - AvgVelocity: {avgVelocity}, AvgPosition: {avgPosition}, AvgLocalOrder: {avgLocalOrder}");

        metricsRecorder.AddMetrics(avgVelocity, avgPosition, avgLocalOrder, Temperature);
    }



    void TrainRidgeRegression()
    {
        float[][] X = metricsRecorder.GetMetrics();
        float[] y = metricsRecorder.GetTemperatures();

        // Calculate and store preprocessing parameters
        CalculatePreprocessingParams(X);

        // Preprocess the features
        float[][] preprocessedX = PreprocessFeatures(X, true);

        // Train the ridge regression model
        ridgeRegression.Train(preprocessedX, y);
    }

    void CalculatePreprocessingParams(float[][] data)
    {
        int nFeatures = data[0].Length;
        featureMeans = new float[nFeatures];
        featureStdDevs = new float[nFeatures];

        for (int feature = 0; feature < nFeatures; feature++)
        {
            featureMeans[feature] = data.Average(sample => sample[feature]);
            featureStdDevs[feature] = Mathf.Sqrt(data.Average(sample => Mathf.Pow(sample[feature] - featureMeans[feature], 2)));
        }
    }

    float[][] PreprocessFeatures(float[][] inputData, bool isTraining)
    {
        int nFeatures = inputData[0].Length;
        int nSamples = inputData.Length;
        float[][] processedData = new float[nSamples][];

        for (int i = 0; i < nSamples; i++)
        {
            processedData[i] = new float[nFeatures];
        }

        for (int feature = 0; feature < nFeatures; feature++)
        {
            float mean = isTraining ? inputData.Average(sample => sample[feature]) : featureMeans[feature];
            float stdDev = isTraining ? Mathf.Sqrt(inputData.Average(sample => Mathf.Pow(sample[feature] - mean, 2))) : featureStdDevs[feature];

            if (isTraining)
            {
                featureMeans[feature] = mean;
                featureStdDevs[feature] = stdDev;
            }

            for (int sample = 0; sample < nSamples; sample++)
            {
                float zScore = (inputData[sample][feature] - mean) / (stdDev + 1e-6f);
                processedData[sample][feature] = Mathf.Clamp(zScore, -3f, 3f);
            }
        }

        return processedData;
    }



    public static float[] NormalizeData(float[] inputData)
    {
        float min = inputData.Min();
        float max = inputData.Max();

        float[] normalizedData = new float[inputData.Length];
        for (int i = 0; i < inputData.Length; i++)
        {
            normalizedData[i] = 2 * (inputData[i] - min) / (max - min);
        }

        return normalizedData;
    }

    public static float[][] NormalizeFeatures(float[][] inputData)
    {
        int nFeatures = inputData[0].Length;
        int nSamples = inputData.Length;
        float[][] normalizedData = new float[nSamples][];

        for (int i = 0; i < nSamples; i++)
        {
            normalizedData[i] = new float[nFeatures];
        }

        for (int feature = 0; feature < nFeatures; feature++)
        {
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int sample = 0; sample < nSamples; sample++)
            {
                float value = inputData[sample][feature];
                if (value < min) min = value;
                if (value > max) max = value;
            }

            float range = max - min;
            for (int sample = 0; sample < nSamples; sample++)
            {
                normalizedData[sample][feature] = range > 1e-6f ? (inputData[sample][feature] - min) / range : 0f;
            }
        }

        return normalizedData;
    }



    private bool simulationComplete = false;

    private RidgeRegression ridgeRegression = new RidgeRegression(0.1f); // Lambda value
    private bool isTraining = true;



    void Start()
    {

        BoidMaterial = new Material(BoidMaterial);
        
        _drawArgsBuffer = new ComputeBuffer(
            1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments
        );

        _drawArgsBuffer.SetData(new uint[5] {
            BoidMesh.GetIndexCount(0), (uint) BoidsCount, 0, 0, 0
        });

        this.boidsData = new GPUBoid[this.BoidsCount];
        this.kernelHandle = _ComputeFlock.FindKernel("CSMain");

        for (int i = 0; i < this.BoidsCount; i++)
            this.boidsData[i] = this.CreateBoidData();

        BoidBuffer = new ComputeBuffer(BoidsCount, Marshal.SizeOf(typeof(GPUBoid)));
        BoidBuffer.SetData(this.boidsData);

        GenerateSkinnedAnimationForGPUBuffer();

        if (UseAffectors) {
            if (UseMeshAffectors) {
                var bounds = MeshAffectors.bounds;
                var scaledVertices = MeshAffectors.vertices.Select(v => (v) * (ReverseYAxisDrawingAffectors ? -1 : 1)  * ScaleDrawingAffectors + DrawingAffectorsOffset).ToArray();
                GenerateDrawingAffectors(scaledVertices, 0, 0, 3);
            }
            else {
                var dataToPaths = new PointsFromData();
                dataToPaths.GeneratePointsFrom(DrawingAffectors, DrawingAffectorsOffset, new Vector3(0, 90, 0), ReverseYAxisDrawingAffectors, ScaleDrawingAffectors);
                GenerateDrawingAffectors(dataToPaths.Points.ToArray());
            }
        }
        else
            AffectorBuffer = new ComputeBuffer(1, Marshal.SizeOf(typeof(GPUAffector)));


        avgVelocityData = new float[1];
        avgPositionData = new Vector3[1];
        localOrderData = new float[1];
        

        // Initialize metric buffers
        avgVelocityBuffer = new ComputeBuffer(1, sizeof(float));
        avgPositionBuffer = new ComputeBuffer(1, sizeof(float) * 3);
        localOrderBuffer = new ComputeBuffer(1, sizeof(float));

        // Bind all buffers to the compute shader
        _ComputeFlock.SetBuffer(kernelHandle, "boidBuffer", BoidBuffer);
        _ComputeFlock.SetBuffer(kernelHandle, "affectorBuffer", AffectorBuffer);

        _ComputeFlock.SetBuffer(kernelHandle, "sharedVelocityBuffer", avgVelocityBuffer);
        _ComputeFlock.SetBuffer(kernelHandle, "sharedPositionBuffer", avgPositionBuffer);
        _ComputeFlock.SetBuffer(kernelHandle, "sharedOrderBuffer", localOrderBuffer);

        outputSumPositionBuffer = new ComputeBuffer(1, sizeof(uint) * 3);
        outputSumVelocityBuffer = new ComputeBuffer(1, sizeof(uint));
        outputSumOrderBuffer = new ComputeBuffer(1, sizeof(uint));
        outputTotalBoidsBuffer = new ComputeBuffer(1, sizeof(uint));

        _ComputeFlock.SetBuffer(kernelHandle, "outputSumPosition", outputSumPositionBuffer);
        _ComputeFlock.SetBuffer(kernelHandle, "outputSumVelocity", outputSumVelocityBuffer);
        _ComputeFlock.SetBuffer(kernelHandle, "outputSumOrder", outputSumOrderBuffer);
        _ComputeFlock.SetBuffer(kernelHandle, "outputTotalBoids", outputTotalBoidsBuffer);

        // Initialize temperature data
        LoadTemperatureData();
        // Normalize the data to be between 0 and 2
        float[] normalizedTemperatureData = NormalizeData(temperatureData);

        // Split the normalized data into training and test sets
        DataSplitter.SplitData(normalizedTemperatureData, out trainData, out testData);
        //DataSplitter.SplitData(temperatureData, out trainData, out testData);



        SetComputeData();
        SetMaterialData();

        if (DrawILoveUnity)
            StartCoroutine(DrawILoveUnityForever());

    }

    void LoadTemperatureData()
    {
        if (TemperatureDataFile != null)
        {
            string[] lines = TemperatureDataFile.text.Split(new char[] { '\n' }, System.StringSplitOptions.RemoveEmptyEntries);
            temperatureData = new float[lines.Length];
            for (int i = 0; i < lines.Length; i++)
            {
                double temp;
                if (double.TryParse(lines[i], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out temp))
                {
                    temperatureData[i] = (float)temp;
                }
                else
                {
                    Debug.LogError($"Failed to parse temperature value at line {i}: '{lines[i]}'");
                }
            }
        }
        else
        {
            Debug.LogError("Temperature data file not assigned in the Inspector!");
        }
    }

    public bool DrawILoveUnity = false;
    public TextAsset EyeDrawing;
    public TextAsset HeartDrawing;
    public TextAsset UnityDrawing;
    IEnumerator DrawILoveUnityForever() {
        var dataToPaths = new PointsFromData();
        dataToPaths.GeneratePointsFrom(EyeDrawing, new Vector3(0, 2, -2), new Vector3(0, 90, 0), ReverseYAxisDrawingAffectors, 0.03f);
        var eyePoints = dataToPaths.Points.ToArray();
        dataToPaths.GeneratePointsFrom(HeartDrawing, new Vector3(0, 2, -2), new Vector3(0, 90, 0), ReverseYAxisDrawingAffectors, 0.05f);
        var heartPoints = dataToPaths.Points.ToArray();
        dataToPaths.GeneratePointsFrom(UnityDrawing, new Vector3(0, 0, -1), new Vector3(0, 90, 0), ReverseYAxisDrawingAffectors, 0.1f);
        var unityPoints = dataToPaths.Points.ToArray();
        yield return new WaitForSeconds(3f);
        while (true) {
            GenerateDrawingAffectors(eyePoints, 0, 0, 0);
            yield return new WaitForSeconds(3f);
            GenerateDrawingAffectors(new Vector3[1], 0, 0, 0);
            yield return new WaitForSeconds(0.5f);
            GenerateDrawingAffectors(heartPoints, 0, 0, 0);
            yield return new WaitForSeconds(3f);
            GenerateDrawingAffectors(new Vector3[1], 0, 0, 0);
            yield return new WaitForSeconds(0.5f);
            GenerateDrawingAffectors(unityPoints, 2, 0, 0);
            yield return new WaitForSeconds(4f);
            GenerateDrawingAffectors(new Vector3[1], 0, 0, 0);
            yield return new WaitForSeconds(2f);
        }
    }

    GPUBoid CreateBoidData()
    {
        GPUBoid boidData = new GPUBoid();
        Vector3 pos = transform.position + Random.insideUnitSphere * SpawnRadius;
        Quaternion rot = Quaternion.Slerp(transform.rotation, Random.rotation, 0.3f);
        boidData.position = pos;
        boidData.direction = rot.eulerAngles;
        boidData.noise_offset = Random.value * 1000.0f;
        boidData.size = Random.Range(0.5f, 1.5f);

        return boidData;
    }

    private void GenerateDrawingAffectors(Vector3[] points, float force = 0, float distance = 0, int axis = 0) {
        if (AffectorBuffer != null)
            AffectorBuffer.Release();

        NbAffectors = points.Length;
        System.Array.Resize(ref Affectors, NbAffectors);

        Affectors = points.Select(p => {
            var affector = new GPUAffector();
            affector.position = p;
            affector.force = force;
            affector.distance = distance;
            affector.axis = axis;
            return affector;
        }).ToArray();

        if (DrawDrawingAffectors) {
            foreach(var point in points) {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.transform.localScale = new Vector3(1,1,1);
                go.transform.position = point;
            }
        }

        AffectorBuffer = new ComputeBuffer(NbAffectors, Marshal.SizeOf(typeof(GPUAffector)));
        AffectorBuffer.SetData(Affectors);
    }

    void SetComputeData() {
        _ComputeFlock.SetFloat("DeltaTime", Time.deltaTime);
        _ComputeFlock.SetFloat("RotationSpeed", RotationSpeed);
        _ComputeFlock.SetFloat("BoidSpeed", BoidSpeed);
        _ComputeFlock.SetFloat("BoidSpeedVariation", BoidSpeedVariation);
        _ComputeFlock.SetVector("FlockPosition", Target.transform.position);
        _ComputeFlock.SetFloat("NeighbourDistance", NeighbourDistance);
        _ComputeFlock.SetFloat("BoidFrameSpeed", BoidFrameSpeed);
        _ComputeFlock.SetInt("BoidsCount", BoidsCount);
        _ComputeFlock.SetInt("NbFrames", NbFramesInAnimation);
        _ComputeFlock.SetInt("NbAffectors", NbAffectors);
        _ComputeFlock.SetFloat("AffectorForce", AffectorForce);
        _ComputeFlock.SetFloat("AffectorDistance", AffectorDistance);
        _ComputeFlock.SetFloat("MaxAffectorFullAxisSize", MaxAffectorFullAxisSize);
        _ComputeFlock.SetInt("StepBoidCheckNeighbours", StepBoidCheckNeighbours);
        _ComputeFlock.SetBuffer(this.kernelHandle, "boidBuffer", BoidBuffer);
        _ComputeFlock.SetBuffer(this.kernelHandle, "affectorBuffer", AffectorBuffer);
        _ComputeFlock.SetFloat("Temperature", Temperature);
    }

    void SetMaterialData() {
        BoidMaterial.SetBuffer("boidBuffer", BoidBuffer);

        if (FrameInterpolation && !BoidMaterial.IsKeywordEnabled("FRAME_INTERPOLATION"))
            BoidMaterial.EnableKeyword("FRAME_INTERPOLATION");
        if (!FrameInterpolation && BoidMaterial.IsKeywordEnabled("FRAME_INTERPOLATION"))
            BoidMaterial.DisableKeyword("FRAME_INTERPOLATION");

        BoidMaterial.SetInt("NbFrames", NbFramesInAnimation);
    }

    // Execution order should be the lowest possible
    void ResetSimulationState()
    {
        // Reset any necessary state variables or accumulators
        currentTemperatureIndex = 0;
        metricsRecorder.Clear();
        // Any other state resets needed
    }

    void Update()
    {
        if (isWarmedUp)
        {
            // 1. Update temperature based on current phase (training or testing)
            if (isTraining)
            {
                if (UseTemperature && trainData != null && currentTemperatureIndex < trainData.Length)
                {
                    Temperature = Mathf.Clamp(trainData[currentTemperatureIndex], 0.1f, 10.0f);
                    Temperature = Temperature * temperatureInputScale;
                    currentTemperatureIndex++;
                }
                else
                {
                    Temperature = 1.0f; // Default to neutral
                }
            }
            else // Testing phase
            {
                if (UseTemperature && testData != null && currentTemperatureIndex < testData.Length)
                {
                    Temperature = Mathf.Clamp(testData[currentTemperatureIndex], 0.1f, 10.0f);
                    Temperature = Temperature * temperatureInputScale;
                    currentTemperatureIndex++;
                }
                else
                {
                    Temperature = 1.0f;
                }
            }
        }

        
        // 2. Set compute shader parameters (including the updated Temperature)
        SetComputeData();
        SetMaterialData();

        _ComputeFlock.SetFloat("Temperature", Temperature);

        // 3. Dispatch the compute shader
        _ComputeFlock.Dispatch(kernelHandle, Mathf.CeilToInt((float)BoidsCount / THREAD_GROUP_SIZE), 1, 1);

        currentFrame++;

        if (!isWarmedUp)
        {
            if (currentFrame >= warmupPeriod)
            {
                isWarmedUp = true;
                Debug.Log("Warmup period completed. Starting actual simulation.");
                // Optionally, reset any accumulators or state here
                ResetSimulationState();
            }
            return; // Skip the rest of the update during warmup
        }

        // 4. Read back the computed metrics
        outputSumPositionBuffer.GetData(sumPositionData);
        outputSumVelocityBuffer.GetData(sumVelocityData);
        outputSumOrderBuffer.GetData(sumOrderData);
        outputTotalBoidsBuffer.GetData(totalBoidsData);

        // 5. Calculate averages
        float totalBoids = totalBoidsData[0];
        Vector3 avgPosition = new Vector3(
            sumPositionData[0] / (SCALE_FACTOR * totalBoids),
            sumPositionData[1] / (SCALE_FACTOR * totalBoids),
            sumPositionData[2] / (SCALE_FACTOR * totalBoids)
        );
        float avgVelocity = sumVelocityData[0] / (SCALE_FACTOR * totalBoids);
        float avgLocalOrder = sumOrderData[0] / (SCALE_FACTOR * totalBoids);

        // 6. Store metrics
        StoreMetrics(avgVelocity, avgPosition, avgLocalOrder);

        // 7. Training or prediction logic
        if (isTraining)
        {
            if (currentTemperatureIndex >= trainData.Length)
            {
                TrainRidgeRegression();
                isTraining = false; // Switch to testing phase
                currentTemperatureIndex = 0; // Reset index for testing
                Debug.Log("Training complete. Moving to testing phase.");
            }
        }
        else // Testing phase
        {
            if (currentTemperatureIndex < testData.Length)
            {
                // Predict using the trained model
                float[] currentMetrics = new float[]
                {
                avgVelocity,
                avgPosition.x,
                avgPosition.y,
                avgPosition.z,
                avgLocalOrder,
                };

                //Debug.Log($"Metrics for Prediction: {string.Join(", ", currentMetrics)}");

                float[][] preprocessedMetrics = PreprocessFeatures(new float[][] { currentMetrics }, false);
                float prediction = ridgeRegression.Predict(preprocessedMetrics[0]);
                predictions.Add(prediction);

                //Debug.Log($"Prediction: {prediction}");
            }
            else if (currentTemperatureIndex >= testData.Length)
            {
                SavePredictions("PredictionsOutput.csv");
                simulationComplete = true;
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#else
                Application.Quit();
#endif
            }
        }

        // 8. Reset buffers for next frame
        outputSumPositionBuffer.SetData(new uint[] { 0, 0, 0 });
        outputSumVelocityBuffer.SetData(new uint[] { 0 });
        outputSumOrderBuffer.SetData(new uint[] { 0 });
        outputTotalBoidsBuffer.SetData(new uint[] { 0 });
    }

    void SavePredictions(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("Prediction");

            foreach (float prediction in predictions)
            {
                writer.WriteLine(prediction.ToString("F6"));
            }
        }
    }





    // Execution order should be the highest possible
    void LateUpdate() {
        Graphics.DrawMeshInstancedIndirect(BoidMesh, 0, BoidMaterial, InfiniteBounds, _drawArgsBuffer, 0);
    }

    void OnDestroy()
    {
        if (BoidBuffer != null) BoidBuffer.Release();
        if (AffectorBuffer != null) AffectorBuffer.Release();
        if (_drawArgsBuffer != null) _drawArgsBuffer.Release();
        if (VertexAnimationBuffer != null) VertexAnimationBuffer.Release();

        if (avgVelocityBuffer != null) avgVelocityBuffer.Release();
        if (avgPositionBuffer != null) avgPositionBuffer.Release();
        if (localOrderBuffer != null) localOrderBuffer.Release();

        if (outputSumPositionBuffer != null) outputSumPositionBuffer.Release();
        if (outputSumVelocityBuffer != null) outputSumVelocityBuffer.Release();
        if (outputSumOrderBuffer != null) outputSumOrderBuffer.Release();
        if (outputTotalBoidsBuffer != null) outputTotalBoidsBuffer.Release();

        if (simulationComplete) // Ensure file is only saved when the simulation completes
        {
            metricsRecorder.SaveMetrics("MetricsOutput.csv");
        }
    }

    private void GenerateSkinnedAnimationForGPUBuffer()
    {
        if (_AnimationClip == null) {
            CreateOneFrameAnimationData();
            return;
        }

        BoidSMR = TargetBoidToGPUSkin.GetComponentInChildren<SkinnedMeshRenderer>();
        _Animator = TargetBoidToGPUSkin.GetComponentInChildren<Animator>();
        int iLayer = 0;
        AnimatorStateInfo aniStateInfo = _Animator.GetCurrentAnimatorStateInfo(iLayer);

        Mesh bakedMesh = new Mesh();
        float sampleTime = 0;
        float perFrameTime = 0;

        NbFramesInAnimation = Mathf.ClosestPowerOfTwo((int)(_AnimationClip.frameRate * _AnimationClip.length));
        perFrameTime = _AnimationClip.length / NbFramesInAnimation;

        var vertexCount = BoidSMR.sharedMesh.vertexCount;
        VertexAnimationBuffer = new ComputeBuffer(vertexCount * NbFramesInAnimation, Marshal.SizeOf(typeof(Vector4)));
        Vector4[] vertexAnimationData = new Vector4[vertexCount * NbFramesInAnimation];
        for (int i = 0; i < NbFramesInAnimation; i++)
        {
            _Animator.Play(aniStateInfo.shortNameHash, iLayer, sampleTime);
            _Animator.Update(0f);

            BoidSMR.BakeMesh(bakedMesh);

            for(int j = 0; j < vertexCount; j++)
            {
                Vector3 vertex = bakedMesh.vertices[j];
                vertexAnimationData[(j * NbFramesInAnimation) +  i] = vertex;
            }

            sampleTime += perFrameTime;
        }

        VertexAnimationBuffer.SetData(vertexAnimationData);
        BoidMaterial.SetBuffer("vertexAnimation", VertexAnimationBuffer);

        TargetBoidToGPUSkin.SetActive(false);
    }

    private void CreateOneFrameAnimationData() {
        var vertexCount = BoidMesh.vertexCount;
        NbFramesInAnimation = 1;
        Vector4[] vertexAnimationData = new Vector4[vertexCount * NbFramesInAnimation];
        VertexAnimationBuffer = new ComputeBuffer(vertexCount * NbFramesInAnimation, Marshal.SizeOf(typeof(Vector4)));
        for(int j = 0; j < vertexCount; j++)
            vertexAnimationData[(j * NbFramesInAnimation)] = BoidMesh.vertices[j];

        VertexAnimationBuffer.SetData(vertexAnimationData);
        BoidMaterial.SetBuffer("vertexAnimation", VertexAnimationBuffer);
        TargetBoidToGPUSkin.SetActive(false);
    }
}