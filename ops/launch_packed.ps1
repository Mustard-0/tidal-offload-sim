$ErrorActionPreference='Stop'
$root='F:\stin_project\tmapo_highload_20261027'
$out=Join-Path $root 'runs\highload_cloud_01'
if(Test-Path -LiteralPath (Join-Path $out 'owner.lock')){throw 'Existing owner lock: no duplicate launch'}
$old=Get-CimInstance Win32_Process -Filter 'ProcessId = 104192'
if($old){throw 'Old owner still present: inspect identity, do not terminate'}
$activation=Join-Path $root ('evidence\packed_activation_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $activation | Out-Null
$env:PYTHONPATH='F:\stin_project\experiment_runtime\lr_crypto_20261026'
$process=Start-Process -FilePath 'D:\anaconda\envs\stin\python.exe' -ArgumentList @(
    '-u',(Join-Path $root 'ops\packed_owner.py'),'--out',$out) -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $activation 'owner.stdout.log') `
    -RedirectStandardError (Join-Path $activation 'owner.stderr.log') -PassThru
@{pid=$process.Id;run=$out;activation=$activation;launched_at=(Get-Date).ToString('o');
  mode='audited adoption; up to20 standard CPU jobs x4 independent processes; unchanged science'} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $activation 'launcher.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $activation 'launcher.json')
