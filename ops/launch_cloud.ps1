$ErrorActionPreference='Stop'
$root='F:\stin_project\tmapo_highload_20261027'
$out=Join-Path $root 'runs\highload_cloud_01'
if(Test-Path -LiteralPath $out){throw 'Existing cloud owner directory: inspect state; never duplicate launch'}
$activation=Join-Path $root ('evidence\cloud_activation_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $activation | Out-Null
$env:PYTHONPATH='F:\stin_project\experiment_runtime\lr_crypto_20261026'
$process=Start-Process -FilePath 'D:\anaconda\envs\stin\python.exe' -ArgumentList @(
    '-u',(Join-Path $root 'ops\cloud_owner.py'),'--out',$out) -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $activation 'owner.stdout.log') `
    -RedirectStandardError (Join-Path $activation 'owner.stderr.log') -PassThru
@{pid=$process.Id;run=$out;activation=$activation;launched_at=(Get-Date).ToString('o');
  mode='bounded standard public CPU; authorized encrypted simulation transfer'} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $activation 'launcher.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $activation 'launcher.json')
