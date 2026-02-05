CREATE UNIQUE INDEX "rocpd_api_ops_api_id_op_id_e7e52317_uniq" ON "rocpd_api_ops" ("api_id", "op_id");
CREATE INDEX "rocpd_api_ops_api_id_f87632ad" ON "rocpd_api_ops" ("api_id");
CREATE INDEX "rocpd_api_ops_op_id_b35ab7c9" ON "rocpd_api_ops" ("op_id");
--CREATE INDEX "rocpd_strin_string_c7b9cd_idx" ON "rocpd_string" ("string");
CREATE INDEX "rocpd_op_description_id_c8dc8310" ON "rocpd_op" ("description_id");
CREATE INDEX "rocpd_op_opType_id_810aaccd" ON "rocpd_op" ("opType_id");
CREATE INDEX "rocpd_counter_op_id_0b4d4b59" ON "rocpd_counter" ("op_id");
CREATE INDEX "rocpd_counter_name_id_92b0ed0c" ON "rocpd_counter" ("name_id");
--CREATE INDEX "rocpd_ustri_string_84d4a7_idx" ON "rocpd_ustring" ("string");
CREATE INDEX "rocpd_api_apiName_id_54bc0a6c" ON "rocpd_api" ("apiName_id");
CREATE INDEX "rocpd_api_category_id_16543984" ON "rocpd_api" ("category_id");
CREATE INDEX "rocpd_api_domain_id_081456b2" ON "rocpd_api" ("domain_id");
CREATE INDEX "rocpd_api_args_id_6a626626" ON "rocpd_api" ("args_id");
CREATE INDEX "rocpd_kernelapi_kernelName_id_c8efa6f0" ON "rocpd_kernelapi" ("kernelName_id");

