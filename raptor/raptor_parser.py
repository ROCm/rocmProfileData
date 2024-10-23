import sys
import pathlib
from typing import List,Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import scipy.stats
import sqlite3
import os

@dataclass
class RaptorParser:
    # usage_doc is also shown in the raptor.py script usage:
    usage_doc = __doc__ = \
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

    # Adding additional info for the doc-string:
    __doc__ += \
    """
    Pandas dataframes:
    (if not specified in the column name, time-based columns use "nano-seconds")

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
    tag : str = None
    category_json : str = None
    gaps : List[int] = None

    roi_start : str = None # string-format fed to make_roi
    roi_end : str = None # string-format fed to make_roi

    top_df : pd.DataFrame = None
    op_df : pd.DataFrame = None

    strings_hash : dict[str] = None
    monitor_df : pd.DataFrame = None

    prekernel_seq : int = 2

    roi_start_ns : int = None
    roi_end_ns   : int = None

    tmp_file : str = None

    # Special internal category names:
    _other_cat = "_Other"
    _gpu_idle_cat = "_GPU_Idle"
    _comm_cat = "_COMM"
    _var_cat = "_Variability"
    

    def __post_init__(self):
        if self.gaps == None:
           self.set_gaps([10])

        if self.category_json is None:
            self.category_json = os.path.join(pathlib.Path(__file__).parent.resolve(), "raptor_cat_vllm.json")

        if os.path.isfile(self.rpd_file):
            import tempfile
            _,extension = os.path.splitext(self.rpd_file)
            if extension == '.gz':
                import gzip
                tmp_path = tempfile.NamedTemporaryFile(delete=True).name
                self.tmp_file = tmp_path
                with gzip.open(self.rpd_file, 'rb') as f_in:
                    with open(tmp_path, "wb") as f_out:
                        f_out.write(f_in.read())
                self.con = sqlite3.connect(tmp_path)
            else:
                self.con = sqlite3.connect(self.rpd_file)
        else:
            raise RuntimeError ("RPD file '" + self.rpd_file + "' does not exist.")

        if self.tag == None:
            self.tag = pathlib.Path(self.rpd_file).stem

        self.first_ns = \
            self.con.execute("select MIN(start) from rocpd_api;").fetchall()[0][0]
        self.last_ns = \
            self.con.execute("select MAX(end) from rocpd_api;").fetchall()[0][0]

        assert self.last_ns >= self.first_ns

        self.set_roi_from_str(self.roi_start, self.roi_end)

    def __del__(self):
        if self.tmp_file:
            try:
                os.remove(self.tmp_file)
            except FileNotFoundError:
                pass
        

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
        self.op_df = self.top_df = self.category_df = None

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

    # translate text tag to number of ns
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
                return match_df.iloc[0][('Start_ns', 'min')]
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

    def get_op_df(self, force=False, kernel_name: str = None):
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
            op_df['Duration'] = np.where(op_df['start'] <= op_df['end'], 
                                         op_df['end'] - op_df['start'], 0)
            op_df.rename(columns={'start' : 'Start_ns', 'end' : 'End_ns',
                                  'description':'Kernel'}, inplace=True)

            # expanding.max computes a running max of end times - so commands that
            # finish out-of-order (with an earlier end) do not move end time.
            op_df['PreGap'] = (op_df['Start_ns'] -  op_df['End_ns'].expanding().max().shift(1)).clip(lower=0)
            op_df.sort_values(['gpuId', 'Start_ns'], ascending=True, inplace=True)
            self.op_df = op_df

        if kernel_name is not None:
            return self.op_df[self.op_df['Kernel'].str.contains(pat=kernel_name, regex=regex)]

        return self.op_df

    def print_op_trace(self, outfile=None, op_df:pd.DataFrame=None,
                       max_ops:int=None, command_print_width=150):
        if op_df is None:
            op_df = self.get_op_df()

        if command_print_width == 0:
            command_print_width = None

        if outfile is None:
            f = sys.stdout
        else:
            f = open(outfile, "w")

        print ("%10s %9s %13s %13s %10s %s" % ("Id", "PreGap_us", "Start_ms", "End_ms", "Dur_us", "Command"), file=f)
        for idx,row in enumerate(op_df.itertuples(),1):
            print ("%10d %9.1f %13s %13s %6.1f %30s" % (row.id,
                                     row.PreGap/1000,
                                     self.pretty_ts(row.Start_ns),
                                     self.pretty_ts(row.End_ns),
                                     row.Duration/1000,
                                     row.Kernel[:command_print_width]
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

    @staticmethod
    def zscore_filter(series):
        """
        Return a series filter which is true for elements who's zscore is <=3,
        ie the values within 3 standard deviations of the mean.
        """
        return abs(scipy.stats.zscore(series))<=3

    @staticmethod
    def zscore_count_outliers(series):
        return (~RaptorParser.zscore_filter(series)).sum().astype(int)

    @staticmethod
    def zscore_min(series):
        " Remove outliers based on zscore, and return the min of the surviving elements "
        if len(series)==1:
            return series.iloc[0]
        else:
            return np.min(series[abs(scipy.stats.zscore(series))<=3])

    @staticmethod
    def zscore_max(series):
        " Remove outliers based on zscore, and return the max of the surviving elements "
        if len(series)==1:
            return series.iloc[0]
        else:
            return np.max(series[abs(scipy.stats.zscore(series))<=3])

    def get_top_df(self, force: bool = False, prekernel_seq: int = None, zscore: bool = False):

        if self.top_df is None or force or \
                (prekernel_seq is not None and prekernel_seq != self.prekernel_seq):

            op_df = self.get_op_df()

            if prekernel_seq == None:
                prekernel_seq = self.prekernel_seq
            self.prekernel_seq = prekernel_seq

            if prekernel_seq:
                self.kernel_cols = ['Kernel']
                for i in range(prekernel_seq):
                    shift_col_name = "Kernel+%d" % (i+1)
                    self.kernel_cols.append(shift_col_name)
                    op_df[shift_col_name] = op_df['Kernel'].shift(i+1)
                top_gb = op_df.groupby(self.kernel_cols, sort=False)
            else:
                top_gb = op_df.groupby(['Kernel'], sort=False)

            self.top_gb = top_gb

            def mymin(col):
                return np.min(col)

            agg_ops = {
                'Start_ns' : ['min', 'max'],
                'PreGap' : ['sum', 'min', 'mean', 'std', 'max'],
            }
            if zscore:
                agg_ops['Duration'] = ['sum', 'min', 'mean', 'std', 'max']
            else:
                agg_ops['Duration'] = ['sum', 'min', 'mean', 'std', 'max']

            top_df = top_gb.agg(agg_ops)

            top_df['Outliers'] = top_gb.agg({'Duration' : self.zscore_count_outliers})

            #top_df[('Duration','min')] = top_gb.agg({'Duration' : self.zscore_min})
            top_df['zmin'] = top_gb.agg({'Duration' : self.zscore_min})
            top_df['zmax'] = top_gb.agg({'Duration' : self.zscore_max})

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
                        ('Start_ns', 'min'):  'min',
                        ('Start_ns', 'max'):  'max',
                        #'TotalCalls' : 'sum',
                        })
            gaps_df['TotalCalls'] = gaps_gb['TotalCalls'].sum()
            gaps_df.columns = \
                [('Duration', col[1]) if col[0]=='PreGap' else col \
                 for col in gaps_df.columns]

            gaps_df.index = ((idx,) for idx in gaps_df.index)

            if not prekernel_seq:
                top_df.index = ((idx,) for idx in top_df.index)
            top_df = pd.concat([top_df, gaps_df])
            top_df.index.name = "Kernel Sequence"

            total_duration = top_df[('Duration','sum')].sum()
            top_df['PctTotal'] = top_df[('Duration','sum')] / total_duration * 100
            self._assign_categories(top_df=top_df)

            if 1:
                top_df['VarSum'] = \
                    ((top_df[('Duration','mean')] - top_df[('Duration','min')]) * \
                      top_df['TotalCalls']).astype(int)

            top_df.loc[top_df['Category'].isin([self._gpu_idle_cat]),'VarSum'] = np.nan

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
                  [('Start_ns','min')   , "First_ms", 1e6, '{:+.3f}'],
                  [('Start_ns','max')   , "Last_ms", 1e6, '{:+.3f}'],
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

    def get_ops_from_top_row(self, top_row:pd.DataFrame):
        op_df = self.get_op_df()
        kernel_seq = top_row.name
        assert len(self.kernel_cols) == len(kernel_seq)
        filter = (op_df[self.kernel_cols] == \
                        pd.Series(kernel_seq, index=self.kernel_cols)).all(axis=1)
        return op_df[filter]

    def get_monitor_df(self):
        if self.monitor_df is None:
            self.monitor_df = pd.read_sql_query("select deviceId,start,end,value from rocpd_monitor where deviceType=='gpu' and monitorType=='sclk' %s order by start ASC" % self.sql_roi_str(add_where=False), self.con)
        return self.monitor_df

    def set_auto_roi(self):
        self.get_category_df()
        top_df = self.get_top_df()
        top_row = top_df[self.top_df['Category'] != 'GAP'].iloc[0]

        self.set_roi_from_abs_ns(roi_start_ns=top_row[('Start_ns','min')],
                                 roi_end_ns=top_row[('Start_ns','max')])

    @staticmethod
    def read_category_file(category_file):
        import json 
          
        with open(category_file) as f:
            data = f.read() 
          
        cats = json.loads(data) 

        return cats

    def _assign_categories(self, top_df:pd.DataFrame=None,
                          categories:Dict=None):
        """
        Create a "Category" column in top_df and assign category 
        labels based on the regex in the categories dict
        """
        if top_df is None:
            top_df = self.get_top_df()

        read_cat = (categories == None)
        categories = {self._gpu_idle_cat : ["^GAP "]}
        if read_cat:
            categories.update(self.read_category_file(self.category_json))

        # Set top_df.cat.  
        # For overlapping patterns, the LAST one wins
        for category_name,pat_list in categories.items():
            mask = pd.Series(False, index=top_df.index)
            for pat in pat_list:
                mask |= top_df.index.str[0].str.contains(pat=pat, regex=True)
            top_df.loc[mask, 'Category'] = category_name
         
        top_df['Category'] = top_df['Category'].fillna(self._other_cat)

    def get_category_df(self, top_df:pd.DataFrame=None, categories:Dict=None,
                        variability_method=None, duration_units='ms'):
        """
        Summarize top kernels into higher-level, user-specified categories.

        variability_method : 
            None : don't add a row for variability.
            comm : Aggregate _COMM category into the _Variability row.  
            non_comm : Aggregate ~_COMM category into the _Variability row
        """

        if top_df is None:
            top_df = self.get_top_df()

        self._assign_categories(top_df, categories)

        # Create the category db 
        cat_gb = top_df.groupby('Category')
        category_df = pd.DataFrame(cat_gb.size(), columns=['UniqKernels'])
        df = cat_gb.agg({
                ('TotalCalls','') : 'sum',
                ('Duration','sum') : 'sum',
                ('VarSum','') : 'sum'
            }) 
        category_df = pd.concat([category_df, df], axis='columns')

        # rename columns and index:
        total_dur_col = 'TotalDur_' + duration_units
        varsum_col = "VarSum_" + duration_units
        category_df.columns=['UniqKernels', 'TotalCalls',
                              total_dur_col, varsum_col]
        category_df.index.name = None

        total_duration_0 = category_df[total_dur_col].sum()

        if variability_method:
            if variability_method=='comm':
                var_filter = category_df.index == self._comm_cat
            elif variability_method=='non_comm':
                var_filter = category_df.index != self._comm_cat
            else:
                raise RuntimeError("bad variability_method")

            category_df.loc[~var_filter, varsum_col] = 0
            category_df.loc[var_filter, total_dur_col] -= \
                category_df.loc[var_filter, varsum_col] 

            #category_df.loc[len(category_df.index)] = [0,0,var_sum,0,0]
            var_df = pd.DataFrame({
                      'UniqKernels': category_df.loc[var_filter,'UniqKernels'].sum(),
                      'TotalCalls':category_df.loc[var_filter, 'TotalCalls'].sum(),
                      total_dur_col : category_df.loc[var_filter, varsum_col].sum(),
                      varsum_col : np.nan
                      }, 
                      index=[self._var_cat])
            category_df = pd.concat([category_df,var_df], axis='rows')

        # Compute Avg and Pct
        category_df['AvgDur_us'] = category_df[total_dur_col] / \
                                   category_df['TotalCalls'] / 1000
        total_duration = category_df[total_dur_col].sum()

        # check we properly acconted for variability only once
        assert total_duration_0 == total_duration
        category_df['Pct'] = category_df[total_dur_col]/total_duration*100

        # finally, sort and convert to the desired units:
        category_df.sort_values(total_dur_col, ascending=False, inplace=True)
        category_df[total_dur_col] /= self._time_units[duration_units]
        category_df[varsum_col] /= self._time_units[duration_units]

        self.category_df = category_df
        return category_df

    def get_variability_df(self, top_df: pd.DataFrame=None, categories: Dict=None):
        top_df = self.get_top_df()
        total_ns = top_df[('Duration','sum')].sum()
        comm_filter = top_df['Category'] == self._comm_cat
        comm_sum = top_df.loc[comm_filter,'VarSum'].sum()
        non_comm_sum = top_df.loc[~comm_filter,'VarSum'].sum()

        comm_dict = {'VarUs'    : [comm_sum/1000, non_comm_sum/1000],
                     'VarPct'   : ["{0:.1%}".format(comm_sum/total_ns), "{0:.1%}".format(non_comm_sum/total_ns)]
                    }
        var_df = pd.DataFrame(comm_dict, index=['by_comm', 'by_non_comm'])
        self.variability_df = var_df
        return var_df
