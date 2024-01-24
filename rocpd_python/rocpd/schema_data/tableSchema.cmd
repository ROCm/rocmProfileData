CREATE TABLE IF NOT EXISTS "rocpd_kernelcodeobject" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "vgpr" integer NOT NULL, "sgpr" integer NOT NULL, "fbar" integer NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_string" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_barrierop" ("op_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, "signalCount" integer NOT NULL, "aquireFence" varchar(8) NOT NULL, "releaseFence" varchar(8) NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_copyapi" ("api_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "size" integer NOT NULL, "width" integer NOT NULL, "height" integer NOT NULL, "kind" integer NOT NULL, "dst" varchar(18) NOT NULL, "src" varchar(18) NOT NULL, "dstDevice" integer NOT NULL, "srcDevice" integer NOT NULL, "sync" bool NOT NULL, "pinned" bool NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_op_inputSignals" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "from_op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, "to_op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_op" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "completionSignal" varchar(18) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_api" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_kernelapi" ("api_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "gridX" integer NOT NULL, "gridY" integer NOT NULL, "gridZ" integer NOT NULL, "workgroupX" integer NOT NULL, "workgroupY" integer NOT NULL, "workgroupZ" integer NOT NULL, "groupSegmentSize" integer NOT NULL, "privateSegmentSize" integer NOT NULL, "kernelArgAddress" varchar(18) NOT NULL, "aquireFence" varchar(8) NOT NULL, "releaseFence" varchar(8) NOT NULL, "codeObject_id" integer NOT NULL REFERENCES "rocpd_kernelcodeobject" ("id") DEFERRABLE INITIALLY DEFERRED, "kernelName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "rocpd_metadata" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "tag" varchar(4096) NOT NULL, "value" varchar(4096) NOT NULL);
CREATE TABLE IF NOT EXISTS "rocpd_monitor" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "deviceType" varchar(16) NOT NULL, "deviceId" integer NOT NULL, "monitorType" varchar(16) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "value" varchar(255) NOT NULL);


INSERT INTO "rocpd_metadata"(tag, value) VALUES ("schema_version", "2")

--CREATE INDEX "rocpd_strin_string_c7b9cd_idx" ON "rocpd_string" ("string");
--CREATE INDEX "rocpd_api_apiName_id_54bc0a6c" ON "rocpd_api" ("apiName_id");
--CREATE INDEX "rocpd_api_args_id_6a626626" ON "rocpd_api" ("args_id");
--CREATE UNIQUE INDEX "rocpd_api_ops_api_id_op_id_e7e52317_uniq" ON "rocpd_api_ops" ("api_id", "op_id");
--CREATE INDEX "rocpd_api_ops_api_id_f87632ad" ON "rocpd_api_ops" ("api_id");
--CREATE INDEX "rocpd_api_ops_op_id_b35ab7c9" ON "rocpd_api_ops" ("op_id");
--CREATE INDEX "rocpd_kernelop_codeObject_id_ad04be1d" ON "rocpd_kernelop" ("codeObject_id");
--CREATE INDEX "rocpd_kernelop_kernelName_id_96546171" ON "rocpd_kernelop" ("kernelName_id");
--CREATE UNIQUE INDEX "rocpd_op_inputSignals_from_op_id_to_op_id_163a30a7_uniq" ON "rocpd_op_inputSignals" ("from_op_id", "to_op_id");
--CREATE INDEX "rocpd_op_inputSignals_from_op_id_5fd8f825" ON "rocpd_op_inputSignals" ("from_op_id");
--CREATE INDEX "rocpd_op_inputSignals_to_op_id_d34a7779" ON "rocpd_op_inputSignals" ("to_op_id");
--CREATE INDEX "rocpd_op_description_id_c8dc8310" ON "rocpd_op" ("description_id");
--CREATE INDEX "rocpd_op_opType_id_810aaccd" ON "rocpd_op" ("opType_id");

