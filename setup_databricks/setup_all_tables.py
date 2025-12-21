# Databricks notebook source
spark.sql("""
    CREATE TABLE IF NOT EXISTS raw_documents (
        id STRING,
        title STRING,
        content STRING,
        url STRING,
        source STRING,
        scraped_date TIMESTAMP,
        content_length INT,
        created_at TIMESTAMP
    )
    USING DELTA
""")

print("✅ Created raw_documents table")
spark.sql("DESCRIBE raw_documents").display()

# COMMAND ----------

spark.sql("""
    CREATE TABLE IF NOT EXISTS processed_chunks (
        chunk_id STRING,
        document_id STRING,
        chunk_text STRING,
        chunk_number INT,
        topic STRING,
        difficulty_level STRING,
        content_type STRING,
        keywords STRING,
        text_length INT,
        word_count INT,
        created_at TIMESTAMP
    )
    USING DELTA
""")

print("✅ Created processed_chunks table")
spark.sql("DESCRIBE processed_chunks").display()

# COMMAND ----------

spark.sql("""
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id STRING,
        embedding ARRAY<FLOAT>,
        model_name STRING,
        embedding_dimension INT,
        created_at TIMESTAMP
    )
    USING DELTA
""")

print("✅ Created chunk_embeddings table")
spark.sql("DESCRIBE chunk_embeddings").display()

# COMMAND ----------

# Databricks notebook source

# MAGIC %md
# MAGIC # Create upload_logs Table

spark.sql("""
    CREATE TABLE IF NOT EXISTS upload_logs (
        run_id STRING,
        pipeline_stage STRING,
        status STRING,
        message STRING,
        records_processed INT,
        duration_seconds INT,
        timestamp TIMESTAMP
    )
    USING DELTA
""")

print("✅ Created upload_logs table")
spark.sql("DESCRIBE upload_logs").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Verify All Tables Created

spark.sql("""
    SELECT 
        'raw_documents' as table_name,
        COUNT(*) as row_count
    FROM raw_documents

    UNION ALL

    SELECT 
        'processed_chunks' as table_name,
        COUNT(*) as row_count
    FROM processed_chunks

    UNION ALL

    SELECT 
        'chunk_embeddings' as table_name,
        COUNT(*) as row_count
    FROM chunk_embeddings

    UNION ALL

    SELECT 
        'upload_logs' as table_name,
        COUNT(*) as row_count
    FROM upload_logs
""").display()

print("✅ All tables created successfully!")