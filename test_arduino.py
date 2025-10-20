import serial
import time

def send_to_arduino(ser, message):
    """Encodes and sends a message to the Arduino with a newline character."""
    # The '|' character is used as a newline separator on the Arduino
    print(f"Sending to LCD: {message.replace('|', ' | ')}")
    ser.write((message + '\n').encode('utf-8'))
    # Wait a bit for the message to be displayed
    time.sleep(2) 

def main():
    """
    Main function to establish serial connection and send test messages.
    """
    try:
        # Establish serial connection.
        # The port is typically '/dev/ttyACM0' for the first connected Arduino.
        # If this fails, you can find the port with `dmesg | grep tty` after plugging it in.
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        
        # It's good practice to wait for the connection to establish.
        # The Arduino resets when a serial connection is opened.
        time.sleep(2)
        print("Successfully connected to Arduino.")
        
        # --- Test Messages ---

        # Test 1: Simple one-line message
        send_to_arduino(ser, "Connection OK!")

        # Test 2: A two-line message using the '|' separator
        send_to_arduino(ser, "Status:|System Ready")

        # Test 3: Another two-line message
        send_to_arduino(ser, "Welcome, Tyler|Initializing...")
        
        print("Test complete. Closing connection.")

    except serial.SerialException as e:
        print(f"Error: Could not open serial port. {e}")
        print("Please check the port name (e.g., '/dev/ttyACM0') and permissions.")
        
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()

