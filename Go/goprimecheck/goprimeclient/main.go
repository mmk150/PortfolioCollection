package main

import (
	"fmt"
	"goprimeclient/lib"
	"io"
	"math/big"
	"net"
	"runtime/debug"
	"strconv"
	"strings"
)

// Function to get the MAC address of the machine
// We'll be lazy and use this for a machine instead of something more complicated
// Rehaul if actually using this project in a real world environment.
func getMACAddress() (string, error) {
	interfaces, err := net.Interfaces()
	if err != nil {
		return "", err
	}

	var macAddresses []string
	for _, i := range interfaces {
		if i.Flags&net.FlagLoopback == 0 && i.Flags&net.FlagUp != 0 {
			mac := i.HardwareAddr.String()
			if mac != "" {
				macAddresses = append(macAddresses, mac)
			}
		}
	}

	if len(macAddresses) > 0 {
		return macAddresses[0], nil
	}
	return "", fmt.Errorf("no MAC address found")
}

func main() {

	// Set memory limit by adjusting garbage collection
	debug.SetMemoryLimit(100 * 1024 * 1024) // 100MB
	debug.SetGCPercent(20)

	macAddress, err := getMACAddress()
	if err != nil {
		fmt.Println("Error getting MAC address:", err)
		return
	}

	//Sim multiclients
	for i := 0; i < 6; i++ {

		newMacAddr := strings.Replace(macAddress, ":", "zz:", i)

		conn, err := net.Dial("tcp", "localhost:8080")
		if err != nil {
			fmt.Println(err)
			return
		}
		defer conn.Close()

		message := fmt.Sprintf("MAC Address: %s | Message: Hello, server!", newMacAddr)

		_, err = conn.Write([]byte(message))
		if err != nil {
			fmt.Println(err)
			return
		}

		fmt.Printf("Sent message: %s\n", message)

		buf := make([]byte, 1024*1024)
		n, err := conn.Read(buf)
		if err != nil {
			if err != io.EOF {
				fmt.Println("Error reading response:", err)
			}
			return
		}

		response := string(buf[:n])
		fmt.Printf("Received response from server: %s\n", response)

		bigint := new(big.Int)
		bigint.SetString(response, 10)

		result := lib.MillerRabin(bigint, 100)
		minusOne := big.NewInt(-1)

		//Not implemented
		if bigint.Cmp(minusOne) == 0 {
			fmt.Println("Server terminated the connection")
			return
		}

		// Send result back to server
		_, err = conn.Write([]byte(strconv.FormatBool(result)))
		if err != nil {
			fmt.Println("Error sending result back to server:", err)
			return
		}
		fmt.Printf("Sent result back to server: %v\n", result)
	}

}
