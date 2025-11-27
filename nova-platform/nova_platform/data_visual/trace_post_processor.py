from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig


class BossaNovaTraceProcessor(TraceProcessor):
    def __init__(self, trace_path):
        super().__init__(trace=trace_path)

    def run_sql(self, sql):
        qr_it = self.query(sql)
        return qr_it

    def get_esl_bw_stat(self):
        sql = """
with xpu_track as (
    SELECT name, id FROM track where name like '%%:dataflow'
),
temp1 as (
    select 
    track_id,ts as leading_ts,
    dur as leading_dur, 
    LEAD(ts) OVER (PARTITION BY track_id ORDER BY ts) AS data_ts,
    LEAD(dur) OVER (PARTITION BY track_id ORDER BY ts) AS data_dur,
    LEAD(arg_set_id) OVER (PARTITION BY track_id ORDER BY ts) AS arg_set_id,
    arg_set_id
    from slice 
    where track_id in (select id from xpu_track) and name like '%%->esl->%%'
    order by track_id, ts
),
rst1 as (
        select leading_ts,leading_dur,data_ts,data_dur, display_value as bytes, display_value/(leading_dur+data_dur) as bw
        from temp1 
        left join args on temp1.arg_set_id=args.arg_set_id and flat_key='debug.total_count'
        where data_dur is not null
)
        select max(bw) as max_bw,min(bw) as min_bw,avg(bw) as avg_bw,count(1) as total_count from rst1 
"""
        qr = self.run_sql(sql)
        return qr
