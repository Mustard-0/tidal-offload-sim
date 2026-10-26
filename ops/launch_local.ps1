$ErrorActionPreference='Stop'
$root='F:\stin_project\tmapo_highload_20261027'
$out=Join-Path $root 'runs\highload_local_01'
if(Test-Path -LiteralPath $out){throw 'Existing owner directory: inspect PID/state; no duplicate start'}
$activation=Join-Path $root ('evidence\activation_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $activation | Out-Null
$env:PYTHONPATH='F:\stin_project\experiment_runtime\torch_cpu_2_5_1;F:\stin_project\tmapo_lr_speed_20261026\.runtime'
$env:GITHUB_SHA=(& git -C $root rev-parse HEAD)
if($LASTEXITCODE -ne 0){throw 'Commit the verified source before formal execution'}
$process=Start-Process -FilePath 'D:\anaconda\envs\stin\python.exe' -ArgumentList @(
    '-u',(Join-Path $root 'ops\local_owner.py'),'--out',$out,'--first-only') -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $activation 'owner.stdout.log') `
    -RedirectStandardError (Join-Path $activation 'owner.stderr.log') -PassThru
@{pid=$process.Id;root=$root;run=$out;activation=$activation;source_commit=$env:GITHUB_SHA;
  mode='local first formal cell, waiting cloud ciphertext authorization';launched_at=(Get-Date).ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $activation 'launcher.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $activation 'launcher.json')
