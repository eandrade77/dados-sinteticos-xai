using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRotator : MonoBehaviour {

    [Header("Resolution Settings")]
    public int imageWidth = 1080;
    public int imageHeight = 720;

    [Header("Rotation Settings")]
    [Tooltip("Degree increment for each capture step (e.g. 5, 10, 15 degrees)")]
    public float angleStep = 10f;
    
    [Tooltip("Time to wait in seconds after rotating before taking the shot (helps animations/physics settle)")]
    public float settleTime = 0.1f;

    [Header("Automation Control")]
    public bool startOnPlay = false;

    private void Start() {
        if (startOnPlay) {
            StartDatasetGeneration();
        }
    }

    // Call this method to start the automation process
    public void StartDatasetGeneration() {
        StartCoroutine(GenerateDatasetRoutine());
    }

    private IEnumerator GenerateDatasetRoutine() {
        Debug.Log("Starting Automated Dataset Generation...");
        float totalRotated = 0f;
        
        while (totalRotated < 360f) {
            // 1. Rotate the camera parent or object around the Y axis
            transform.Rotate(0, angleStep, 0);
            totalRotated += angleStep;

            // 2. Wait for physics or animations to settle if needed
            if (settleTime > 0) {
                yield return new WaitForSeconds(settleTime);
            }

            // 3. Wait until the very end of the frame (post-render) so we capture clean frames
            yield return new WaitForEndOfFrame();

            // 4. Trigger the static screenshot method
            ScreenshotHandler.TakeScreenshot_Static(imageWidth, imageHeight);
            
            Debug.Log(string.Format("Captured frame at angle: {0:F1}°", totalRotated));
        }

        Debug.Log("Automated Dataset Generation Completed successfully for 360 degrees!");
    }
}
