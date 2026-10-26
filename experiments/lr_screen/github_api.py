"""GitHub transport using existing Git credential manager; never logs tokens."""
import json
import os
import subprocess
import urllib.error
import urllib.request
import urllib.parse

REPOSITORY = 'Mustard-0/tidal-offload-sim'


class GitHub:
    def __init__(self, token=None):
        if token is not None:
            self.token = token
            return
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
        raw = subprocess.check_output(['git','credential','fill'],
            input='protocol=https\nhost=github.com\n\n', text=True, env=env,
            stderr=subprocess.DEVNULL, timeout=30)
        fields = dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        self.token = fields['password']

    def request(self, method, path, value=None):
        if not path.startswith('/repos/'+REPOSITORY+'/') and path != '/repos/'+REPOSITORY:
            raise ValueError('Unexpected repository API target')
        data = None if value is None else json.dumps(value).encode()
        req = urllib.request.Request('https://api.github.com'+path, data=data, method=method,
            headers={'Authorization':'Bearer '+self.token, 'Accept':'application/vnd.github+json',
                     'X-GitHub-Api-Version':'2022-11-28','User-Agent':'lr-screen-coordinator'})
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
        return json.loads(payload) if payload else None

    def upload(self, release_id, name, data):
        url = ('https://uploads.github.com/repos/'+REPOSITORY+'/releases/'+str(int(release_id))+
               '/assets?name='+urllib.parse.quote(name,safe=''))
        req = urllib.request.Request(url,data=data,method='POST',headers={
            'Authorization':'Bearer '+self.token,'Content-Type':'application/octet-stream',
            'User-Agent':'lr-screen-coordinator'})
        with urllib.request.urlopen(req,timeout=300) as response:
            return json.loads(response.read())


def download(asset):
    url = asset['browser_download_url']
    if not url.startswith('https://github.com/'+REPOSITORY+'/releases/download/'):
        raise ValueError('Unexpected asset download origin')
    # Public ciphertext download: never send an account token to redirects.
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'lr-screen-coordinator'}),timeout=300) as f:
        data = f.read()
    if len(data) != asset['size']:
        raise ValueError('GitHub asset size mismatch')
    from .transport import sha
    if asset.get('digest') and asset['digest'] != 'sha256:'+sha(data):
        raise ValueError('GitHub asset digest mismatch')
    return data


if __name__ == '__main__':
    api = GitHub()
    repo = api.request('GET','/repos/'+REPOSITORY)
    policy = api.request('GET','/repos/'+REPOSITORY+'/actions/permissions')
    print(json.dumps(dict(repository=repo['full_name'],visibility=repo['visibility'],
                          actions_enabled=policy.get('enabled'),authenticated=True)))
