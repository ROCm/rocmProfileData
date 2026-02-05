CREATE TABLE IF NOT EXISTS "rocpd_metadata" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "tag" varchar(4096) NOT NULL, "value" varchar(4096) NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_string" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_ustring" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);

CREATE TABLE IF NOT EXISTS "rocpd_api" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "category_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "domain_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" bigint NOT NULL REFERENCES "rocpd_ustring" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_copyapi" ("api_ptr_id" bigint NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "size" integer NOT NULL, "width" integer NOT NULL, "height" integer NOT NULL, "kind" integer NOT NULL, "dst" varchar(18) NOT NULL, "src" varchar(18) NOT NULL, "dstDevice" integer NOT NULL, "srcDevice" integer NOT NULL, "sync" bool NOT NULL, "pinned" bool NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_kernelapi" ("api_ptr_id" bigint NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "gridX" integer NOT NULL, "gridY" integer NOT NULL, "gridZ" integer NOT NULL, "workgroupX" integer NOT NULL, "workgroupY" integer NOT NULL, "workgroupZ" integer NOT NULL, "groupSegmentSize" integer NOT NULL, "privateSegmentSize" integer NOT NULL, "kernelName_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);


CREATE TABLE IF NOT EXISTS "rocpd_op" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" bigint NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" bigint NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);


CREATE TABLE IF NOT EXISTS "rocpd_monitor" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "deviceType" varchar(16) NOT NULL, "deviceId" integer NOT NULL, "monitorType" varchar(16) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "value" varchar(255) NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_counter" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "value" real NOT NULL, "op_id" bigint NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, "name_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_stackframe" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_ptr_id" bigint NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "depth" integer NOT NULL, "name_id" bigint NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);

INSERT INTO "rocpd_metadata"(tag, value) VALUES ("gpu_stride", "1000");
INSERT INTO "rocpd_metadata"(tag, value) VALUES ("pid_stride", "10000000");
INSERT INTO "rocpd_metadata"(tag, value) VALUES ("schema_version", "3");

--CREATE UNIQUE INDEX "rocpd_api_ops_api_id_op_id_e7e52317_uniq" ON "rocpd_api_ops" ("api_id", "op_id");
--CREATE INDEX "rocpd_api_ops_api_id_f87632ad" ON "rocpd_api_ops" ("api_id");
--CREATE INDEX "rocpd_api_ops_op_id_b35ab7c9" ON "rocpd_api_ops" ("op_id");
--CREATE INDEX "rocpd_strin_string_c7b9cd_idx" ON "rocpd_string" ("string");
--CREATE INDEX "rocpd_op_description_id_c8dc8310" ON "rocpd_op" ("description_id");
--CREATE INDEX "rocpd_op_opType_id_810aaccd" ON "rocpd_op" ("opType_id");
--CREATE INDEX "rocpd_counter_op_id_0b4d4b59" ON "rocpd_counter" ("op_id");
--CREATE INDEX "rocpd_counter_name_id_92b0ed0c" ON "rocpd_counter" ("name_id");
--CREATE INDEX "rocpd_ustri_string_84d4a7_idx" ON "rocpd_ustring" ("string");
--CREATE INDEX "rocpd_api_apiName_id_54bc0a6c" ON "rocpd_api" ("apiName_id");
--CREATE INDEX "rocpd_api_category_id_16543984" ON "rocpd_api" ("category_id");
--CREATE INDEX "rocpd_api_domain_id_081456b2" ON "rocpd_api" ("domain_id");
--CREATE INDEX "rocpd_api_args_id_6a626626" ON "rocpd_api" ("args_id");
--CREATE INDEX "rocpd_kernelapi_kernelName_id_c8efa6f0" ON "rocpd_kernelapi" ("kernelName_id");
