using CTRE.Phoenix;
using CTRE.Phoenix.Controller;
using CTRE.Phoenix.MotorControl;
using CTRE.Phoenix.MotorControl.CAN;
using Microsoft.SPOT;
using System;
using System.IO.Ports;
using System.Text;
using System.Threading;

namespace HERO_UART_Smooth_Control
{
    public class Program
    {
        static TalonSRX rightSlave = new TalonSRX(4);
        static TalonSRX right = new TalonSRX(3);
        static TalonSRX leftSlave = new TalonSRX(2);
        static TalonSRX left = new TalonSRX(1);

        static StringBuilder uartBuffer = new StringBuilder();
        static GameController _gamepad = null;
        static SerialPort _uart = null;

        static float targetV = 0.0f;
        static float targetW = 0.0f;
        static float smoothedV = 0.0f;
        static float smoothedW = 0.0f;

        static float alpha = 0.6f;              // smoothing factor
        static float maxChangePerLoop = 0.15f;  // velocity ramp limit

        public static void Main()
        {
            byte[] data = new byte[1024];
            int bytesReceived = 0;

            bool useUART = false;
            bool lastButton6State = false;

            while (true)
            {
                if (_gamepad == null)
                    _gamepad = new GameController(UsbHostDevice.GetInstance());

                // Toggle control mode on Button 6 press
                bool currentButton6State = _gamepad.GetButton(6);
                if (currentButton6State && !lastButton6State)
                {
                    useUART = !useUART;
                    Debug.Print("Mode toggled: " + (useUART ? "UART" : "Joystick"));

                    if (useUART)
                    {
                        if (_uart == null)
                        {
                            _uart = new SerialPort(CTRE.HERO.IO.Port1.UART, 115200);
                            _uart.Open();
                            uartBuffer.Clear();
                        }
                    }
                    else
                    {
                        if (_uart != null)
                        {
                            _uart.Flush();
                            _uart.Close();
                            _uart = null;
                        }
                    }
                }
                lastButton6State = currentButton6State;

                if (!useUART)
                {
                    float v = -1 * _gamepad.GetAxis(1);
                    float w = _gamepad.GetAxis(2);
                    Drive(v, w, false);
                }
                else
                {
                    if (_uart != null && _uart.BytesToRead > 0)
                    {
                        bytesReceived = _uart.Read(data, 0, data.Length);
                        for (int i = 0; i < bytesReceived; i++)
                        {
                            char c = (char)data[i];
                            uartBuffer.Append(c);

                            if (c == '#')
                            {
                                string msg = uartBuffer.ToString();
                                uartBuffer.Clear();

                                try
                                {
                                    int startIdx = msg.IndexOf('!');
                                    int midIdx = msg.IndexOf('@');
                                    int endIdx = msg.IndexOf('#');

                                    if (startIdx != -1 && midIdx != -1 && endIdx != -1 &&
                                        startIdx < midIdx && midIdx < endIdx &&
                                        endIdx > midIdx + 1 && midIdx > startIdx + 1)
                                    {
                                        string vStr = msg.Substring(startIdx + 1, midIdx - (startIdx + 1));
                                        string wStr = msg.Substring(midIdx + 1, endIdx - (midIdx + 1));

                                        float v = (float)Convert.ToDouble(vStr);
                                        float w = (float)Convert.ToDouble(wStr);

                                        targetV = LimitChange(targetV, v, maxChangePerLoop);
                                        targetW = LimitChange(targetW, w, maxChangePerLoop);

                                        Debug.Print("Parsed v=" + v + ", w=" + w);
                                    }
                            }

                            if (uartBuffer.Length > 100)
                                uartBuffer.Clear();
                        }
                    }

                    smoothedV = alpha * targetV + (1 - alpha) * smoothedV;
                    smoothedW = alpha * targetW + (1 - alpha) * smoothedW;

                    Drive(smoothedV, smoothedW, true);
                }

                CTRE.Phoenix.Watchdog.Feed();
                Thread.Sleep(20);  // 50Hz update rate
            }
        }

        static void Drive(float v, float w, bool isUART)
        {
            if (isUART)
                v = -v;  // Only invert v for UART (not for joystick)

            float leftThrot = v + w;
            float rightThrot = v - w;

            leftThrot = Clamp(leftThrot, -1.0f, 1.0f);
            rightThrot = Clamp(rightThrot, -1.0f, 1.0f);

            left.Set(ControlMode.PercentOutput, leftThrot);
            leftSlave.Set(ControlMode.PercentOutput, leftThrot);
            right.Set(ControlMode.PercentOutput, -rightThrot);
            rightSlave.Set(ControlMode.PercentOutput, -rightThrot);
        }

        static float Clamp(float value, float min, float max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        static float LimitChange(float current, float target, float maxDelta)
        {
            float delta = target - current;
            if (delta > maxDelta) return current + maxDelta;
            if (delta < -maxDelta) return current - maxDelta;
            return target;
        }
    }
}
