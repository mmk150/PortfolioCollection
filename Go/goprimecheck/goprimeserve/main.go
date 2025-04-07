package main

import (
	"database/sql"
	"fmt"
	"goprimeserve/lib"
	"log"
	"net"
	"strings"
)

// "math/big"
// "strconv"

func dbendio(db *lib.Database, mac string, prime_candidate string, result bool) (string, error) {
	id, workingOn, err := db.GetUserByAddress(mac)
	if err != nil {
		return "", fmt.Errorf("error finding user with MAC %s: %v", mac, err)
	}

	if workingOn != prime_candidate {
		return "", fmt.Errorf("user %s is not assigned to work on %s (currently working on: %s)",
			mac, prime_candidate, workingOn)
	}

	isPrimeStr := false
	if result {
		isPrimeStr = true
	}

	db.UpdateCheckedStatus(prime_candidate, "TRUE")
	db.UpdateIsPrimeStatus(prime_candidate, isPrimeStr)

	db.ClearUserWorkingOn(id)

	if result {
		log.Printf("Number %s confirmed as prime by user %s\n\n", prime_candidate, mac)
	} else {
		log.Printf("Number %s confirmed as not prime by user %s\n\n", prime_candidate, mac)
	}

	return "NONE", nil
}

func dbio(db *lib.Database, mac string) (string, error) {
	id, workingOn, err := db.GetUserByAddress(mac)

	if err == sql.ErrNoRows {
		log.Printf("New MAC address detected: %s", mac)
		err = db.InsertUser(mac, "NONE")
		if err != nil {
			return "", fmt.Errorf("error inserting new user with MAC %s: %v", mac, err)
		}
		return dbio(db, mac)
	} else if err != nil {
		return "", fmt.Errorf("database error checking MAC %s: %v", mac, err)
	}

	if workingOn != "NONE" {
		return workingOn, nil
	}

	unchecked, err := db.GetUncheckedEntries()
	if err != nil {
		return "", fmt.Errorf("error getting unchecked entries: %v", err)
	}

	if len(unchecked) == 0 {
		db.AddNewPossiblePrimes(10)
		return dbio(db, mac)
	}

	nextNumber := unchecked[0]

	err = db.UpdateCheckedStatus(nextNumber, "IN_PROGRESS")
	if err != nil {
		return "", fmt.Errorf("error updating checked status: %v", err)
	}

	err = db.UpdateUserWorkingOn(id, nextNumber)
	if err != nil {
		return "", fmt.Errorf("error updating user working status: %v", err)
	}

	return nextNumber, nil
}

func main() {

	db, err := lib.NewDatabase("./mydb.sqlite")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	ln, err := net.Listen("tcp", ":8080")

	if err != nil {
		fmt.Println(err)
		return
	}

	for {
		conn, err := ln.Accept()
		if err != nil {
			fmt.Println(err)
			continue
		}
		go handleConnection(db, conn)
	}

}

func handleConnection(db *lib.Database, conn net.Conn) {

	defer conn.Close()

	remoteAddr := conn.RemoteAddr().String()
	host, port, _ := net.SplitHostPort(remoteAddr)

	fmt.Printf("New connection from:\n")
	fmt.Printf("Remote Address: %s\n", remoteAddr)
	fmt.Printf("Client IP: %s\n", host)
	fmt.Printf("Client Port: %s\n\n", port)

	names, err := net.LookupAddr(host)
	if err == nil && len(names) > 0 {
		fmt.Printf("Hostname: %s\n", names[0])
	}

	buf := make([]byte, 1024)
	n, err := conn.Read(buf)

	if err != nil {
		fmt.Println(err)
		return
	}
	message := string(buf[:n])
	fmt.Printf("Received from %s: %s \n\n", remoteAddr, message)

	var conn_mac_addr string
	parts := strings.Split(message, "|")
	if len(parts) >= 1 {
		mac_part := strings.TrimSpace(parts[0])
		if strings.HasPrefix(mac_part, "MAC Address:") {
			conn_mac_addr = strings.TrimSpace(strings.TrimPrefix(mac_part, "MAC Address:"))
		}
	}

	var response_to_conn string
	response_to_conn, err = dbio(db, conn_mac_addr)
	db.AddNewPossiblePrimes(2)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("Sending to %s: %s \n", remoteAddr, response_to_conn)
	_, err = conn.Write([]byte(response_to_conn))
	if err != nil {
		fmt.Println("Error sending response:", err)
		return
	}

	n, err = conn.Read(buf)

	var mbool bool

	if err != nil {
		fmt.Println(err)
		return
	}
	message = string(buf[:n])
	if message == "false" {
		mbool = false
	} else {
		mbool = true
	}
	fmt.Printf("Received from %s: %s \n", remoteAddr, message)
	dbendio(db, conn_mac_addr, response_to_conn, mbool)

}
