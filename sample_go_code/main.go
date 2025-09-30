package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, World!")
    })

    log.Println("Server starting on :8080")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatal(err)
    }
}

func processData(data []string) ([]string, error) {
    if data == nil {
        return nil, fmt.Errorf("data cannot be nil")
    }

    result := make([]string, 0, len(data))
    for _, item := range data {
        if item != "" {
            result = append(result, item)
        }
    }

    return result, nil
}

type Config struct {
    Port     int    `json:"port"`
    Host     string `json:"host"`
    Database string `json:"database"`
}

func NewConfig() *Config {
    return &Config{
        Port:     8080,
        Host:     "localhost",
        Database: "app.db",
    }
}

func (c *Config) Validate() error {
    if c.Port <= 0 {
        return fmt.Errorf("port must be positive")
    }
    if c.Host == "" {
        return fmt.Errorf("host cannot be empty")
    }
    return nil
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        handleGet(w, r)
    case http.MethodPost:
        handlePost(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func handleGet(w http.ResponseWriter, r *http.Request) {
    data := map[string]interface{}{
        "status": "ok",
        "time":   time.Now(),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(data)
}

func handlePost(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Message string `json:"message"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    log.Printf("Received message: %s", req.Message)
    fmt.Fprintf(w, "Message received: %s", req.Message)
}