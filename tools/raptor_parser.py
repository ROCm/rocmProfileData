import sys
from typing import List,Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sqlite3
import os

@dataclass
class RaptorParser:
    """
    Tool for parsing and post-processing RPD files:
    - slice, dice, and eviscerate RPD files into focused regions-of-interest.
    - generate summaries of top kernels and combine into categories (ie GEMM, Comm, Attention, etc).
    - compute “Gaps” where GPU is idle.
    - compute kernel to kernel variability and percentage of execution time
    - auto-roi feature to focus on hottest region
    - tables in the RaptorParser class are organized into Pandas dataframes for interactive analysis via ipython/jupyter/etc.
    - show a text trace of each command's execution - kernel name, duration, idle gaps, etc
    - RAPTOR possibly stands for ROCm Profile-Data Tool and Output Refinement.
    """

    """
    Pandas dataframes:
    (if not specified in the column name, columns use "nano-seconds")

    op_df       : Record for each GPU operation (ie kernel).  Includes pre-gap, duration,
                  call count, name, etc.
    top_df      : Group ops with same name into a summary.  Add time for Gaps.
    category_df : Combine rows from top_df into user-specified categories.  
                  Add "Other" category for kernels not matching any of the patterns.
    
    pretty_top_df : Convert a subset of the top_df patterns into formatted columns - 
                    for example converting timestamps from raw ns to a ms value 
                    that is relative to the start of the trace.  Designed to be easily viewed
                    on the screen to spot trends.
    """

    rpd_file : str = None
    category_json : str = None
    gaps : List[int] = None

    roi_start : str = None # string-format fed to make_roi
    roi_end : str = None # string-format fed to make_roi

    top_df : pd.DataFrame = None
    op_df : pd.DataFrame = None

    op_trace_df : pd.DataFrame = None

    strings_hash : dict[str] = None
    monitor_df : pd.DataFrame = None

    roi_start_ns : int = None
    roi_end_ns   : int = None

    def __post_init__(self):
        if self.gaps == None:
           self.set_gaps([10])

        if self.category_json is None:
            import pathlib
            self.category_json = os.path.join(pathlib.Path(__file__).parent.resolve(), "raptor_cat_vllm.json")

        if os.path.isfile(self.rpd_file):
            self.con = sqlite3.connect(self.rpd_file)
        else:
            raise RuntimeError ("RPD file '" + self.rpd_file + "' does not exist.")

        self.first_ns = \
            self.con.execute("select MIN(start) from rocpd_api;").fetchall()[0][0]
        self.last_ns = \
            self.con.execute("select MAX(end) from rocpd_api;").fetchall()[0][0]

        assert self.last_ns >= self.first_ns

        self.set_roi_from_str(self.roi_start, self.roi_end)

    def set_gaps(self, gaps):
        self.gaps = gaps
        self.gaps.sort()
        self.gaps = [0] + self.gaps + [np.inf]

    def set_roi_from_abs_ns(self, roi_start_ns, roi_end_ns):
        self.roi_start_ns = roi_start_ns
        self.roi_end_ns   = roi_end_ns

        assert self.roi_start_ns >= 0
        assert self.roi_start_ns <= self.last_ns - self.first_ns
        assert self.roi_end_ns >= 0
        assert self.roi_end_ns <= self.last_ns - self.first_ns
        assert self.roi_start_ns <= self.roi_end_ns

        # prevent mis-use:
        self.roi_start = None
        self.roi_end = None

        # force recomputation:
        self.op_df = self.top_df = self.op_trace_df = self.category_df = None

    def set_roi_from_str(self, roi_start, roi_end=None):
        if roi_start == None:
            roi_start_ns = 0
        else:
            roi_start_ns = self.make_roi(roi_start)

        if roi_end == None:
            roi_end_ns = self.last_ns - self.first_ns
        else:
            roi_end_ns = self.make_roi(roi_end)

        self.set_roi_from_abs_ns(roi_start_ns, roi_end_ns)


    def trim_to_roi(self, new_file_name:str = None, inplace:bool=False):
        if not inplace:
            import pathlib,shutil
            if new_file_name is None:
                new_file_name = pathlib.PurePath(self.rpd_file).with_suffix(".trim.rpd")
                print (f"{new_file_name}")
                shutil.copy2(self.rpd_file, new_file_name)
                
            if os.path.isfile(new_file_name):
                connection = sqlite3.connect(new_file_name)
            else:
                raise RuntimeError ("RPD file '" + new_file_name + "' does not exist.")
        else:
            connection = self.con

        connection.execute("delete from rocpd_api where start < %d or start > %d" % \
                            (self.roi_start_ns, self.roi_end_ns))
        connection.execute("delete from rocpd_api_ops where api_id not in (select id from rocpd_api)")
        connection.execute("delete from rocpd_op where id not in (select op_id from rocpd_api_ops)")
        try:
            connection.execute("delete from rocpd_monitor where start < (select min(start) from rocpd_api) or start > (select max(end) from rocpd_op)")
        except:
            pass

        connection.commit()

        #clear any unused strings
        if 0:
            stringCount = connection.execute("select count(*) from rocpd_string").fetchall()[0][0]
            from rocpd.importer import RocpdImportData
            from rocpd.strings import cleanStrings
            importData = RocpdImportData()
            importData.resumeExisting(connection) # load the current db state
            cleanStrings(importData, False)
            stringRemaingCount = connection.execute("select count(*) from rocpd_string").fetchall()[0][0]
            print(f"Removed {stringCount - stringRemaingCount} of {stringCount} strings.  {stringRemaingCount} remaining")

        connection.isolation_level = None
        connection.execute("vacuum")
        connection.commit()
        connection.close()

    _time_units = {
        "ns"  : 1,
        "us"  : 1000,
        "ms"  : 1000*1000,
        "s"   : 1000*1000*1000,
        "sec" : 1000*1000*1000,
    }

    def parse_roi(self, roi_str, default_unit="ns"):
        picked_unit = default_unit
        if roi_str[0].isdigit():
            for unit in self._time_units.keys():
                if roi_str.endswith(unit):
                    roi_str = roi_str[:-len(picked_unit)]
                    picked_unit = unit
                    break

            scale = self._time_units[picked_unit]
            return int(float(roi_str) * scale)
        else:
            top_df = self.get_top_df()
            match_df = top_df[top_df.index.str.contains(pat=roi_str, regex=True)]
            if len(match_df):
                return match_df.iloc[0][('StartNs', 'min')]
            else:
                raise RuntimeError("ROI string '%s' not found in top_df" % roi_str)

    def make_roi(self, roi: str):
        if "%" in roi:
            time_ns = int( (self.last_ns - self.first_ns) * ( int( roi.replace("%","") )/100 ) + self.first_ns )
        elif roi.startswith("+"):
            time_ns = int(self.parse_roi(roi[1:], default_unit="ms"))
        elif roi.startswith("-"):
            time_ns = int(self.last_ns - self.parse_roi(roi[1:], default_unit="ms"))
        else:
            time_ns = int(self.parse_roi(roi, default_unit="ms"))

        return time_ns

    def pretty_ts(self, timestamp_ns:int, div=None):
        """ Pretty-print the timestmap to show a . separator after the ms boundary """
        if div and timestamp_ns != fp.nan:
            timestamp_ns /= div
        timestamp_ms = timestamp_ns // 1e6
        return "%8d.%06d" % (timestamp_ms, timestamp_ns - timestamp_ms*1e6)

    def print_timestamps(self, indent=""):
        print(indent, "first    :", self.first_ns)
        print(indent, "last     :", self.last_ns)
        print(indent, "Timestamps :  RelTime(ms)")
        print(indent, "  roi_start:", self.pretty_ts(self.roi_start_ns))
        print(indent, "  roi_end  :", self.pretty_ts(self.roi_end_ns))
        print(indent, "  roi_dur  :", self.pretty_ts(self.roi_end_ns - self.roi_start_ns))
        
    def sql_roi_str(self, add_where=True):
        rv = "where " if add_where else ""
        rv += "start >= %d and start <= %d" % (self.roi_start_ns + self.first_ns, 
                                               self.roi_end_ns + self.first_ns)
        return rv

    def get_op_df(self, force=False):
        """ 
        Read the op table from the sql input into op_df.
        Add a PreGap measurement between commands.
        """

        if self.op_df is None or force:
            ops_query = "select * from op %s order by start ASC " % self.sql_roi_str()
            #self.raw_ops_df = pd.read_sql_query(ops_query, self.con) 
            op_df = pd.read_sql_query(ops_query, self.con) 

            op_df = op_df[op_df['opType'].isin(['KernelExecution', 'CopyDeviceToDevice', 'Task'])]

            # normalize timestamps:
            op_df["start"] -= self.first_ns
            op_df["end"]   -= self.first_ns

            # expanding.max computes a running max of end times - so commands that
            # finish out-of-order (with an earlier end) do not move end time.
            op_df['PreGap'] = (op_df['start'] -  op_df['end'].expanding().max().shift(1)).clip(lower=0)
            self.op_df = op_df

        return self.op_df

    def get_op_trace_df(self, force=False, kernel_name:str=None, regex=True):
        """
        Add extra fields to the op_df for pre-gap, frequency, etc
        """ 
        if self.op_trace_df is None or force:
            op_df = self.get_op_df()
            op_df.sort_values(['gpuId', 'start'], ascending=True)

            strings_hash = self.get_strings_hash()

            op_trace_df = pd.DataFrame(index=op_df.index)
            scratch_df = pd.DataFrame(index=op_df.index)

            op_trace_df['id'] = op_df['id']
            op_trace_df['GpuId'] = op_df['gpuId']
            op_trace_df['QueueId'] = op_df['queueId']
            op_trace_df['StartNs'] = op_df['start']
            op_trace_df['EndNs'] = op_df['end']
            op_trace_df['PreGap'] = np.where(op_df['PreGap'] >= 0,
                                                   op_df['PreGap'], np.nan)
            op_trace_df['Duration'] = np.where(op_df['start'] <= op_df['end'], 
                                                     op_df['end'] - op_df['start'], 0)
            if 'opType_id' in op_df.columns:
                op_trace_df['OpType'] = op_df['opType_id'].map(strings_hash)
            elif 'opType' in op_df.columns:
                op_trace_df['OpType'] = op_df['opType']
            else:
                raise RuntimeError("Can't determine optype")

            if 'descripion_id' in op_df.columns:
                op_trace_df['Command'] = np.where(op_trace_df['OpType'].isin(['KernelExecution', 'Task']), 
                                                    op_df['description_id'].map(strings_hash),
                                                    op_trace_df['OpType'])

            elif 'description' in op_df.columns:
                op_trace_df['Command'] = np.where(op_trace_df['OpType'].isin(['KernelExecution', 'Task']), 
                                                    op_df['description'],
                                                    op_trace_df['OpType'])
            self.op_trace_df = op_trace_df

        if kernel_name is not None:
            return self.op_trace_df[self.op_trace_df['Command'].str.contains(pat=kernel_name, regex=regex)]

        return self.op_trace_df

    def print_op_trace(self, outfile=None, op_trace_df:pd.DataFrame=None, max_ops:int=None, command_print_width=150):
        if op_trace_df is None:
            op_trace_df = self.get_op_trace_df()

        if command_print_width == 0:
            command_print_width = None

        if outfile is None:
            f = sys.stdout
        else:
            f = open(outfile, "w")

        print ("%10s %9s %13s %13s %10s %s" % ("Id", "PreGap_us", "Start_ms", "End_ms", "Dur_us", "Command"), file=f)
        for idx,row in enumerate(op_trace_df.itertuples(),1):
            print ("%10d %9.1f %13s %13s %6.1f %30s" % (row.id,
                                     row.PreGap/1000,
                                     self.pretty_ts(row.StartNs),
                                     self.pretty_ts(row.EndNs),
                                     row.Duration/1000,
                                     row.Command[:command_print_width]
                                     ), file=f)
            if max_ops is not None and idx>=max_ops:
                break

        if outfile is not None:
            f.close()

    @staticmethod
    def _make_gaps_labels(gaps : List[int]):
        assert len(gaps) >= 2
        gaps_labels = []
        if len(gaps)>2:
            gaps_labels += ["GAP <%dus" % gaps[1]]
        gaps_labels += ["GAP %dus-%dus" % (gaps[i], gaps[i+1]) for i in range(1, len(gaps)-2)]
        gaps_labels += ["GAP >%dus" % gaps[-2]]

        return gaps_labels

    def get_top_df(self, force:bool = False, leading_kernels=None):

        if self.top_df is None or force:
            op_trace_df = self.get_op_trace_df()
            if leading_kernels:
                op_trace_df['PreCommand'] = op_trace_df['Command'].shift(1)
                top_gb = op_trace_df.groupby(['Command','PreCommand'], sort=False)
            else:
                top_gb = op_trace_df.groupby('Command', sort=False)
            top_df = top_gb.agg({
                'PreGap' : ['sum', 'min', 'mean', 'max'],
                'Duration' : ['sum', 'min', 'mean', 'max'],
                'StartNs' : ['min', 'max']
                 })

            top_df['TotalCalls'] = top_gb.size()

            # Gaps:
            # Extract pre-gap info from each command and create separate rows 
            # in the top_df summary.
            # Multiple gap buckets are supported.
            gaps_gb = top_df.groupby(pd.cut(top_df[('PreGap', 'sum')],
                                    pd.Series(self.gaps)*1000,
                                    labels=self._make_gaps_labels(self.gaps)),
                                    observed=True)
            gaps_df = gaps_gb.agg({
                        ('PreGap', 'sum'):  'sum',
                        ('PreGap', 'min'):  'min',
                        ('PreGap', 'mean'): 'mean',
                        ('PreGap', 'max'):  'max',
                        ('StartNs', 'min'):  'min',
                        ('StartNs', 'max'):  'max',
                        #'TotalCalls' : 'sum',
                        })
            gaps_df['TotalCalls'] = gaps_gb['TotalCalls'].sum()
            gaps_df.columns = \
                [('Duration', col[1]) if col[0]=='PreGap' else col \
                 for col in gaps_df.columns]

            top_df = pd.concat([top_df, gaps_df])

            total_duration = top_df[('Duration','sum')].sum()
            top_df['PctTotal'] = top_df[('Duration','sum')] / total_duration * 100

            # Add the Category column:
            self.get_category_df(top_df=top_df)

            top_df['VarSum'] = \
                ((top_df[('Duration','mean')] - top_df[('Duration','min')]) * \
                  top_df['TotalCalls']).astype(int)

            top_df.loc[top_df['Category'].isin(['GAP']),'VarSum'] = np.nan

            top_df = top_df.sort_values([('Duration', 'sum')],
                                             ascending=False)
            
            self.top_df = top_df

        return self.top_df

    def get_pretty_top_df(self, top_df=None):
        scale = 1000

        if top_df is None:
            top_df = self.get_top_df()
        self.get_category_df(top_df)

        mapper = [
                  # Index : (Remapped-name, scale factor, display format)
                  [('StartNs','min')   , "First_ms", 1e6, '{:+.3f}'],
                  [('StartNs','max')   , "Last_ms", 1e6, '{:+.3f}'],
                  [('PreGap','mean')   , "PreGap_mean_us", scale, '{:.1f}'],
                  [('Duration','min')  , "Dur_min_us", scale, '{:.1f}'],
                  [('Duration','mean') , "Dur_mean_us", scale, '{:.1f}'],
                  [('Duration','max')  , "Dur_max_us", scale, '{:.1f}'],
                  [('VarSum','')  ,      "VarUs", None, None],
                  [('VarSum','')  ,      "VarPct", None, None],
                  [('PctTotal', ''),     "PctTotal", None, '{0:.1f}%'],
                  #[('Duration','sum')  ,("DurSum_us", scale, '{:.0f}')],
                  [('TotalCalls', ''),   "TotalCalls", None, '{0:.0f}'],
                  [('PctTotal', ''),     "PctTotal", None, '{0:.1f}%'],
                  [('Category', ''),     "Category", None, '{0:s}'],
                 ]

        pretty_top_df = pd.DataFrame(index=top_df.index)
        for (top_df_col,pretty_col,scale,fmt_str) in mapper:
            if fmt_str:
                if scale is not None:
                    pretty_top_df[pretty_col] = \
                        (top_df[top_df_col] / scale).apply(fmt_str.format)
                else:
                    pretty_top_df[pretty_col] = top_df[top_df_col].apply(fmt_str.format)
            else:
                pretty_top_df[pretty_col] = top_df[top_df_col]

        pretty_top_df['VarUs']  = top_df['VarSum'] / 10000 / top_df['TotalCalls']
        pretty_top_df['VarPct'] = (top_df['VarSum'] / top_df[('Duration','sum')]).apply("{0:.1%}".format)

        self.pretty_top_df = pretty_top_df

        return pretty_top_df

    def get_strings_hash(self, force=False):
        if self.strings_hash is None or force:
            string_df = pd.read_sql_query("select * from rocpd_string", self.con) 
            #string_df['idx'] = string_df.index
            string_df.index = string_df['id']
            self.strings_hash = string_df['string'].T.to_dict()

            string_df.index = string_df['string']
            self._string_to_id_hash = string_df['id'].T.to_dict()
        return self.strings_hash

    def get_monitor_df(self):
        if self.monitor_df is None:
            self.monitor_df = pd.read_sql_query("select deviceId,start,end,value from rocpd_monitor where deviceType=='gpu' and monitorType=='sclk' %s order by start ASC" % self.sql_roi_str(add_where=False), self.con)
        return self.monitor_df

    def set_auto_roi(self):
        self.get_category_df()
        top_df = self.get_top_df()
        top_row = top_df[self.top_df['Category'] != 'GAP'].iloc[0]

        self.set_roi_from_abs_ns(roi_start_ns=top_row[('StartNs','min')],
                                 roi_end_ns=top_row[('StartNs','max')])

    @staticmethod
    def read_category_file(category_file):
        import json 
          
        with open(category_file) as f:
            data = f.read() 
          
        cats = json.loads(data) 

        return cats

    def get_category_df(self, top_df:pd.DataFrame=None, categories:Dict=None, variability_method=None):
        """
        Summarize top kernels into higher-level, user-specified categories.

        variability_method : None=don't show variability, comm=show $Variability_Comm, non_comm=show $Variability_NonComm
        """


        if top_df is None:
            top_df = self.get_top_df()

        if categories is None:
            categories = self.read_category_file(self.category_json)

        # Set top_df.cat.  
        # For overlapping patterns, the LAST one wins
        for category_name,pat_list in categories.items():
            mask = pd.Series(False, index=top_df.index)
            for pat in pat_list:
                mask |= top_df.index.str.contains(pat=pat, regex=True)
            top_df.loc[mask, 'Category'] = category_name
         
        other_name = "Other"
        top_df['Category'] = top_df['Category'].fillna(other_name)

        # Create the category db 
        cat_gb = top_df.groupby('Category')
        category_df = pd.DataFrame(cat_gb.size(), columns=['UniqKernels'])
        df = cat_gb.agg({
                ('TotalCalls','') : 'sum',
                ('Duration','sum') : 'sum'
            }) 
        category_df = pd.concat([category_df, df], axis='columns')
        category_df.columns=['UniqKernels', 'TotalCalls', 'TotalDuration']
        category_df.index.name = None
        category_df['AvgDuration'] = category_df['TotalDuration'] / category_df['TotalCalls']
        total_duration = category_df['TotalDuration'].sum()
        category_df['Pct'] = category_df['TotalDuration']/total_duration*100
        category_df.sort_values('TotalDuration', ascending=False, inplace=True)

        self.category_df = category_df
        return category_df

    def get_variability_df(self, top_df: pd.DataFrame=None, categories: Dict=None):
        top_df = self.get_top_df()
        total_ns = top_df[('Duration','sum')].sum()
        comm_filter = top_df['Category'] == 'Comm'
        comm_sum = top_df.loc[comm_filter,'VarSum'].sum()
        non_comm_sum = top_df.loc[~comm_filter,'VarSum'].sum()

        comm_dict = {'VarUs'    : [comm_sum/1000, non_comm_sum/1000],
                     'VarPct'   : ["{0:.1%}".format(comm_sum/total_ns), "{0:.1%}".format(non_comm_sum/total_ns)]
                    }
        var_df = pd.DataFrame(comm_dict, index=['by_comm', 'by_non_comm'])
        self.variability_df = var_df
        return var_df
