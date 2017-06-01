using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Collections;
using System.Diagnostics;
using System.Threading;

namespace demo
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        public byte[] readFile(string name)
        {
            FileStream fs = new FileStream(name, FileMode.Open, FileAccess.Read, FileShare.None);
            BinaryReader sr = new BinaryReader(fs);
            byte[] r = sr.ReadBytes((int)fs.Length);
            sr.Close();
            fs.Close();
            return r;
        }

        public string getLabelString(int[] label)
        {
            FileStream fs = new FileStream("码表.txt", FileMode.Open, FileAccess.Read, FileShare.None);
            StreamReader sr = new StreamReader(fs);
            string nextLine;
            ArrayList map = new ArrayList();
            while ((nextLine = sr.ReadLine()) != null)
                map.Add(nextLine);
            
            string ot = "";
            for(int i = 0; i < label.Length; ++i){
                ot = ot + map[label[i]];
            }
            sr.Close();
            fs.Close();
            return ot;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            IntPtr model = CC.createClassifier("deploy.prototxt", "nin_iter_16000.caffemodel");
            byte[] img = readFile("测试图片.png");
            IntPtr softmax = CC.predictSoftmax(model, img, img.Length);
            int[] label = new int[4];

            CC.getMultiLabel(softmax, label);
            MessageBox.Show("识别结果是: " + getLabelString(label));

            CC.releaseSoftmaxResult(softmax);
            CC.releaseClassifier(model);
        }


        public struct TrainValInfo
        {
            public int iterNum;
            public int numOutput;
            public float[] values;
            public string[] outputNames;
            public override string ToString()
            {
                string ot = "iterNum: " + iterNum + ", numOutput: " + numOutput + ", values: [";
                for (int i = 0; i < numOutput; ++i)
                    ot += values[i] + (i == numOutput-1 ? "" : ", ");
                ot += "], outputNames: [";

                for (int i = 0; i < numOutput; ++i)
                    ot += outputNames[i] + (i == numOutput - 1 ? "" : ", ");
                ot += "]";
                return ot;
            }
        }

        public unsafe TrainValInfo getInfo(void* param)
        {
            TrainValInfo info = new TrainValInfo();
            sbyte* p = (sbyte*)param;
            info.iterNum = *(int*)p; p += 4;
            info.numOutput = *(int*)p; p += 4;

            float* values = *(float**)p; p+=sizeof(void*);
            if (info.numOutput > 0)
            {
                info.values = new float[info.numOutput];
                for (int i = 0; i < info.numOutput; ++i)
                    info.values[i] = values[i];

                sbyte** names = *(sbyte***)(p);
                info.outputNames = new string[info.numOutput];
                for (int i = 0; i < info.numOutput; ++i)
                    info.outputNames[i] = new string(names[i]);
            }
            return info;
        }

        //下面是训练的代码
        public unsafe int trainCallback(int eventFlag, int param1, float param2, void* param3)
        {
            if (eventFlag == 3 && param1 > 10)
                return 2;

            if (eventFlag == 7)
            {
                TrainValInfo info = getInfo(param3);
                Debug.WriteLine(info);
            }
            Debug.WriteLine("{0}, {1}, {2}", eventFlag, param1, param2);
            return 0;
        }

        private unsafe void button2_Click(object sender, EventArgs e)
        {
            CC.TraindEventCallback func = new CC.TraindEventCallback(trainCallback);
            CC.setTraindEventCallback(func);

            Thread t = new Thread(new ThreadStart(run));
            t.Start();
        }

        public void run()
        {
            CC.train_network("train --solver=solver.prototxt");
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Thread t = new Thread(new ThreadStart(convertImageSet));
            t.Start();
        }


        const int event_readlabel = 1;
        const int event_shuffle = 2;
        const int event_initdb = 3;
        const int event_put_one = 4;
        const int event_err_one = 5;
        const int event_finish = 6;
        public unsafe int convertImageSetCallback(int eventFlag, int param1, float param2, void* param3)
        {
            switch (eventFlag)
            {
                case event_readlabel:
                    Debug.WriteLine("读了标签文件啦，总共有{0}个", param1);
                    break;

                case event_shuffle:
                    Debug.WriteLine("做了随机打乱了哟~");
                    break;

                case event_initdb:
                    Debug.WriteLine("开始初始化lmdb啦，宽度是{0}，高度是{1}，目录路径是：{2}", param1, (int)param2, new string((sbyte*)param3));
                    break;

                case event_put_one:
                    Debug.WriteLine("处理了一个，共处理了{0}个，当前处理到了第{1}个", param1, (int)param2);
                    break;

                case event_err_one:
                    Debug.WriteLine("错误了一个，已经处理了{0}个，当前处理到了第{1}个，错误的文件路径：{2}", param1, (int)param2, new string((sbyte*)param3));
                    break;

                case event_finish:
                    Debug.WriteLine("转换完毕啦，总共处理了{0}个文件", param1);
                    break;
            }
            //Debug.WriteLine("{0}, {1}, {2}", eventFlag, param1, param2);
            return 0;
        }

        public unsafe void convertImageSet()
        {
            CC.convert_imageset("./ label-train.txt train_lmdb --shuffle=true --resize_width=224 --resize_height=224", new CC.ConvertImageSetEventCallback(convertImageSetCallback));
        }
    }
}
