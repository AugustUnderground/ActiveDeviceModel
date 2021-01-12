#!/usr/bin/env -S julia --project

using Pluto
host = chomp(read(`hostname -I`, String));
port = 8888;
Pluto.run(host = host, port = port, launch_browser = false);
