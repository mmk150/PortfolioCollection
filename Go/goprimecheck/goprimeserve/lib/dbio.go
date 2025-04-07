package lib

import (
	"database/sql"
	"fmt"
	"log"
	"math/big"
	"os"

	_ "github.com/mattn/go-sqlite3"
)

type Database struct {
	db *sql.DB
}

func NewDatabase(dbPath string) (*Database, error) {
	needsInit := !exists(dbPath)

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	err = db.Ping()
	if err != nil {
		return nil, err
	}

	database := &Database{
		db: db,
	}

	if needsInit {
		err = database.Initialize()
		if err != nil {
			return nil, err
		}
	}

	return database, nil
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

func (d *Database) Initialize() error {

	queries := []string{
		`CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        address TEXT NOT NULL,
		working_on TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );`,
		`CREATE TABLE IF NOT EXISTS primes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value TEXT NOT NULL UNIQUE,
        isPrime TEXT NOT NULL,
		checked TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );`,
	}

	for _, query := range queries {
		_, err := d.db.Exec(query)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d *Database) Close() error {
	return d.db.Close()
}

func (d *Database) InsertUser(address, working_on string) error {
	insertSQL := `INSERT INTO users (address, working_on) VALUES (?, ?)`
	_, err := d.db.Exec(insertSQL, address, working_on)
	return err
}

func (d *Database) GetUser(id int) (string, string, error) {
	var address, working_on string
	query := `SELECT address, working_on FROM users WHERE id = ?`
	err := d.db.QueryRow(query, id).Scan(&address, &working_on)
	return address, working_on, err
}

func (d *Database) UpdateUserWorkingOn(id int, working_on string) error {
	updateSQL := `UPDATE users SET working_on = ? WHERE id = ?`
	result, err := d.db.Exec(updateSQL, working_on, id)
	if err != nil {
		return err
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("no user found with id %d", id)
	}

	return nil
}

func (d *Database) GetUserByAddress(address string) (int, string, error) {
	var id int
	var working_on string
	query := `SELECT id, working_on FROM users WHERE address = ?`
	err := d.db.QueryRow(query, address).Scan(&id, &working_on)
	return id, working_on, err
}

func (d *Database) ClearUserWorkingOn(id int) error {
	return d.UpdateUserWorkingOn(id, "NONE")
}

func (d *Database) InsertPrimeNumber(value string) error {
	insertSQL := `INSERT INTO primes (value, isPrime, checked) VALUES (?, "FALSE", "FALSE")`
	_, err := d.db.Exec(insertSQL, value)
	return err
}

func (d *Database) UpdateIsPrimeStatus(value string, isPrime bool) error {
	status := "FALSE"
	if isPrime {
		status = "TRUE"
	}

	updateSQL := `UPDATE primes SET isPrime = ? WHERE value = ?`
	result, err := d.db.Exec(updateSQL, status, value)
	if err != nil {
		return err
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("no entry found with value %s", value)
	}

	return nil
}

func (d *Database) UpdateCheckedStatus(value string, status string) error {
	validStatuses := map[string]bool{
		"TRUE":        true,
		"FALSE":       true,
		"IN_PROGRESS": true,
	}

	if !validStatuses[status] {
		return fmt.Errorf("invalid status: %s. Must be TRUE, FALSE, or IN_PROGRESS", status)
	}

	updateSQL := `UPDATE primes SET checked = ? WHERE value = ?`
	result, err := d.db.Exec(updateSQL, status, value)
	if err != nil {
		return err
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return fmt.Errorf("no entry found with value %s", value)
	}

	return nil
}

func (d *Database) GetPrimeEntry(value string) (string, string, string, error) {
	var isPrime, checked string
	var createdAt string
	query := `SELECT isPrime, checked, created_at FROM primes WHERE value = ?`
	err := d.db.QueryRow(query, value).Scan(&isPrime, &checked, &createdAt)
	return isPrime, checked, createdAt, err
}

func (d *Database) GetUncheckedEntries() ([]string, error) {
	rows, err := d.db.Query(`SELECT value FROM primes WHERE checked = "FALSE"`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var values []string
	for rows.Next() {
		var value string
		if err := rows.Scan(&value); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, nil
}

func (d *Database) GetInProgressEntries() ([]string, error) {
	rows, err := d.db.Query(`SELECT value FROM primes WHERE checked = "IN_PROGRESS"`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var values []string
	for rows.Next() {
		var value string
		if err := rows.Scan(&value); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, nil
}

func (d *Database) AddNewPossiblePrimes(count int) error {
	var startStr string
	err := d.db.QueryRow(`
        SELECT value FROM primes 
        ORDER BY CAST(value AS DECIMAL) DESC 
        LIMIT 1
    `).Scan(&startStr)

	// If no entries exist, start at 479001599
	if err == sql.ErrNoRows {
		startStr = "479001599"
	} else if err != nil {
		return fmt.Errorf("error querying last prime: %v", err)
	}

	start := new(big.Int)
	_, success := start.SetString(startStr, 10)
	if !success {
		return fmt.Errorf("failed to convert %s to big.Int", startStr)
	}

	one := big.NewInt(1)
	two := big.NewInt(2)

	if new(big.Int).Mod(start, two).Cmp(big.NewInt(0)) == 0 {
		start.Add(start, one)
	}

	added := 0
	for added < count {
		start.Add(start, two)
		valueStr := start.String()
		err = d.InsertPrimeNumber(valueStr)

		if err != nil {

			if err.Error() == "UNIQUE constraint failed: primes.value" {
				continue
			}
			return fmt.Errorf("error inserting prime %s: %v", valueStr, err)
		}

		added++
		if added%100 == 0 {
			log.Printf("Added %d numbers. Current number: %s", added, valueStr)
		}
	}

	log.Printf("Successfully added %d new numbers to check", count)
	return nil
}
