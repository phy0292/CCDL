using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace demo
{
    class CC
    {
        [DllImport("classification_dll.dll", EntryPoint = "createClassifier", CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr createClassifier(
            string prototxt_file,
            string caffemodel_file,
            float scale_raw=1,
            string mean_file=null, 
            int num_means = 0,
            float[] means = null,
            int gpu_id = -1);

        [DllImport("classification_dll.dll", EntryPoint = "predictSoftmax", CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr predictSoftmax(
            IntPtr classifier,
            byte[] img,
            int len = 1,
            int top_n = 1);

        [DllImport("classification_dll.dll", EntryPoint = "getMultiLabel", CallingConvention = CallingConvention.StdCall)]
        public static extern void getMultiLabel(
            IntPtr softmax,
            int[] buf);

        [DllImport("classification_dll.dll", EntryPoint = "releaseClassifier", CallingConvention = CallingConvention.StdCall)]
        public static extern void releaseClassifier(IntPtr model);

        [DllImport("classification_dll.dll", EntryPoint = "releaseSoftmaxResult", CallingConvention = CallingConvention.StdCall)]
        public static extern void releaseSoftmaxResult(IntPtr softmax);


        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public unsafe delegate int TraindEventCallback(int eventFlag, int param1, float param2, void* param3);


        [DllImport("classification_dll.dll", EntryPoint = "setTraindEventCallback", CallingConvention = CallingConvention.StdCall)]
        public static extern void setTraindEventCallback(TraindEventCallback callback);

        [DllImport("classification_dll.dll", EntryPoint = "train_network", CallingConvention = CallingConvention.StdCall)]
        public static extern int train_network(string param);
    }
}
