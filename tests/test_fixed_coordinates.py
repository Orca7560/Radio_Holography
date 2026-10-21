import unittest,tempfile,os,csv
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corr_fringe as c
import group_up_txt_prd as g

class FixedCoordinates(unittest.TestCase):
 def test_lag_keeps_nominal_centres(self):
  tasks=c.make_offset_tasks(.5,scan_half=True)
  for lag in [-100,100]:
   adjusted=c.apply_scan_lag(tasks,lag,41000)
   self.assertEqual(len(tasks),len(adjusted))
   for a,b in zip(tasks,adjusted):
    self.assertEqual(a['skip']+a['length']/2,b['nominal_skip']+b['nominal_length']/2)
   self.assertNotEqual(tasks[1]['skip'],adjusted[1]['skip'])
 def test_no_silent_missing_bins(self):
  with self.assertRaises(ValueError):c.apply_scan_lag([{'skip':0,'length':10}],-100,1000)
 def test_pipeline_actual_time_and_fixed_position(self):
  for lag in [0,-100,100]:
   with self.subTest(lag=lag),tempfile.TemporaryDirectory() as tmp:
    old=os.getcwd();os.chdir(tmp)
    try:
     Path('step_cor').mkdir();Path('step_cor/YAMAGU32_YAMAGU34_2026184082203_X.cor').touch()
     Path('test.xml').write_text('<root><stream><label>X</label></stream><process><epoch>2026/184 08:22:03</epoch><skip>0</skip><length>41</length></process></root>')
     data=dict(epoch='',label='X',source='3C273',length=.5,amp=1,snr=10,phase=0,frequency=8448,noise_level=.1,res_rate=0,x_az=0,x_el=0,x_h=0,y_az=0,y_el=0,y_h=0,mjd=0,rfi='-',bp='-',acf='-')
     with patch.object(c,'infer_scan_direction',return_value=1),patch.object(c,'run_external_command',return_value='fake'),patch.object(c,'parse_fringe_output',side_effect=lambda *a,**k:dict(data)):
      c.run_fringe_steps(['test.xml'],processor='frinZ',use_freq_format=True,scan_half=True,scan_lag_ms=lag,scan_positions=[1])
     calls=[]
     def position(records,times,epoch,length,lag,**kwargs):
      calls.append((epoch,length));return (3*((epoch-datetime(2026,7,3,8,22,3)).total_seconds()+length/2),60,3,0)
     with patch.object(g,'parse_prd_positions',return_value=[(datetime(2026,7,3),0,0)]),patch.object(g,'prd_offset_at',side_effect=position):
      g.merge_fringe_results('.',output_file='beam.txt',prd_filepath='dummy')
     rows=list(csv.DictReader(open('beam.txt'),skipinitialspace=True))
     tasks=c.make_offset_tasks(.5,scan_half=True)
     self.assertEqual(len(rows),len(tasks))
     for row,task in zip(rows,tasks):self.assertAlmostEqual(float(row['Az_Offset']),3*(task['skip']+task['length']/2)/1000)
     self.assertEqual(rows[1]['Epoch'],datetime(2026,7,3,8,22,3,int((.25+lag/1000)*1e6)).strftime('%Y/%j %H:%M:%S.%f'))
     if lag==-100:self.assertAlmostEqual(float(rows[0]['Length']),.15)
    finally:os.chdir(old)
if __name__=='__main__':unittest.main()
