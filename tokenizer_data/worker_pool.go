package main

import (
    "context"
    "sync"
    "time"
)

type WorkerPool struct {
    workers   int
    tasks     chan func()
    wg        sync.WaitGroup
    ctx       context.Context
    cancel    context.CancelFunc
}

func NewWorkerPool(workers int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    return &WorkerPool{
        workers: workers,
        tasks:   make(chan func(), workers*2),
        ctx:     ctx,
        cancel:  cancel,
    }
}

func (p *WorkerPool) Start() {
    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go p.worker()
    }
}

func (p *WorkerPool) worker() {
    defer p.wg.Done()

    for {
        select {
        case task := <-p.tasks:
            task()
        case <-p.ctx.Done():
            return
        }
    }
}

func (p *WorkerPool) Submit(task func()) {
    select {
    case p.tasks <- task:
    case <-p.ctx.Done():
    }
}

func (p *WorkerPool) Shutdown(timeout time.Duration) error {
    p.cancel()

    done := make(chan struct{})
    go func() {
        p.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        return nil
    case <-time.After(timeout):
        return fmt.Errorf("shutdown timeout")
    }
}
