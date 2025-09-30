package main

import (
    "flag"
    "fmt"
    "os"
)

type Config struct {
    Host     string
    Port     int
    Verbose  bool
    LogLevel string
}

func main() {
    cfg := &Config{}

    flag.StringVar(&cfg.Host, "host", "localhost", "Server host")
    flag.IntVar(&cfg.Port, "port", 8080, "Server port")
    flag.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")
    flag.StringVar(&cfg.LogLevel, "log-level", "info", "Log level (debug|info|warn|error)")

    flag.Parse()

    if err := cfg.Validate(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }

    if err := run(cfg); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}

func (c *Config) Validate() error {
    if c.Port < 1 || c.Port > 65535 {
        return fmt.Errorf("invalid port: %d", c.Port)
    }

    validLevels := map[string]bool{
        "debug": true, "info": true, "warn": true, "error": true,
    }

    if !validLevels[c.LogLevel] {
        return fmt.Errorf("invalid log level: %s", c.LogLevel)
    }

    return nil
}

func run(cfg *Config) error {
    fmt.Printf("Starting server on %s:%d\n", cfg.Host, cfg.Port)
    // Server implementation here
    return nil
}
