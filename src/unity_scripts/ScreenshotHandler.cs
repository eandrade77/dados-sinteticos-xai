using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class ScreenshotHandler : MonoBehaviour {

    private static ScreenshotHandler instance;

    private Camera myCamera;
    private bool takeScreenshotOnNextFrame;

    [Header("Dataset Settings")]
    [Tooltip("Name of the subfolder for the target class (e.g., Dalmatian, Doberman)")]
    public string classFolderName = "Doberman";

    private void Awake() {
        instance = this;
        myCamera = gameObject.GetComponent<Camera>();
    }

    // Returns formatted filename with timestamp
    private string GetFilename(int width, int height) {
        return string.Format("screen_{0}x{1}_{2}.jpg",
                              width, height,
                              System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss-fff")); // added milliseconds to prevent collisions
    }

    private void OnPostRender() {
        if (takeScreenshotOnNextFrame) {
            takeScreenshotOnNextFrame = false;
            RenderTexture renderTexture = myCamera.targetTexture;

            Texture2D renderResult = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.ARGB32, false);
            Rect rect = new Rect(0, 0, renderTexture.width, renderTexture.height);
            renderResult.ReadPixels(rect, 0, 0);

            byte[] byteArray = renderResult.EncodeToJPG();
            
            // Build path dynamically using the target class subfolder
            string targetDirectory = Path.Combine(Application.dataPath, "Dataset", classFolderName);
            
            // Ensure directory exists
            if (!Directory.Exists(targetDirectory)) {
                Directory.CreateDirectory(targetDirectory);
            }
            
            string fullPath = Path.Combine(targetDirectory, GetFilename(renderTexture.width, renderTexture.height));
            File.WriteAllBytes(fullPath, byteArray);
            Debug.Log("Dataset Image Saved to: " + fullPath);

            RenderTexture.ReleaseTemporary(renderTexture);
            myCamera.targetTexture = null;
        }
    }

    private void TakeScreenshot(int width, int height) {
        myCamera.targetTexture = RenderTexture.GetTemporary(width, height, 16);
        takeScreenshotOnNextFrame = true;
    }

    public static void TakeScreenshot_Static(int width, int height) {
        if (instance != null) {
            instance.TakeScreenshot(width, height);
        } else {
            Debug.LogError("ScreenshotHandler instance is missing in the scene!");
        }
    }
}
