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
    }
}
