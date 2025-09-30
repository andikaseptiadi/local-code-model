package main

import (
    "context"
    "fmt"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
)

type S3Client struct {
    svc *s3.S3
}

func NewS3Client() (*S3Client, error) {
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-west-2"),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    return &S3Client{
        svc: s3.New(sess),
    }, nil
}

func (c *S3Client) UploadFile(ctx context.Context, bucket, key string, data []byte) error {
    _, err := c.svc.PutObjectWithContext(ctx, &s3.PutObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   bytes.NewReader(data),
    })
    return err
}
