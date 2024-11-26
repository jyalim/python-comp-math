---
title: Setup
---

## Overview

This lesson is designed to be run on the Sol supercomputer with a
Jupyter Lab server. This ensures a modern and consistent environment
among attendees. Instructions are given below on how to connect to the
supercomputer and get started.  All of the software and data used in
this lesson are freely available online, and instructions on how to
obtain them are provided within the lesson.

## Connect to the supercomputer

#### First, if off campus, connect to VPN

Attendees that are off campus will need to first connect to [ASU's
virtual private network (VPN)][sslvpn]. If not already installed, use
the previous link, sign in as if signing into MyASU, and follow the
download and installation instructions. To sign into the VPN, connect to
[sslvpn.asu.edu][sslvpn] with the now installed Cisco VPN client. The
resulting prompt requires an asurite, the corresponding password, and a
two factor authentication method (i.e., `push`, `call`, `sms`, or a
six-digit code provided by Duo). The last field may be labeled as
`second password` on some Cisco clients. N.B., if you are on a Mac, some
additional troubleshooting will be required ([fix][sslvpn-mac-fix]).

#### Second, in your preferred browser, connect to supercomputer

The supercomputer's [web portal][web-portal] provides a consistent user
interface across all major operating systems. This fact is leveraged by
these lessons. To connect, go to [sol.asu.edu][web-portal] in your
preferred browser. If the VPN is required, the website will not load.
Otherwise, you will be prompted to sign in as if signing into MyASU.

#### Launch a Jupyter Lab Server

We will be running Python from a modern graphical interface provided by
a Jupyter Lab server. To launch one:

1. From the gold navigation bar at the top of the supercomputer's 
[web portal][web-portal], select, `Interactive Sessions` with your
mouse.
2. From the resulting drop down, select `Jupyter`.
3. On the resulting form, select the 
* `lightwork` partition, 
* `public` 'QOS',
* `1` core,
* `4` GiB of memory,    
and submit the form.
4. Your Jupyter Server should be ready within a minute. Select `Launch`
   on the resulting page.

#### Jupyter Lab quickstart

When you start Jupyter for the first time, you'll be greeted with a 
**file system viewer** on the left-hand side of the screen and a
**launcher** on the right-hand side. To get to the lesson materials, use
the file system viewer: double-click the `Desktop` directory then the
`python-comp-math` directory. Open a notebook called,
`00-quickstart.ipynb`. To evaluate the first and only **cell** in the
new view of the file, use either the "play"-button icon in the menu bar
or use the keyboard shortcut <kbd>shift</kbd>+<kbd>enter</kbd>.

The default cell type is called `Code` and thus typical Jupyter notebook
cells evaluate Python code. However, Jupyter may use arbitrary backends
to run notebook cells which has made it a popular development
environment for remote systems. This lesson will exclusively use Python
`Code` cells, but a second common cell type, `Markdown`, is useful for
providing richly formatted content within a notebook. Both cell types
are demonstrated in the demo notebook, `00-quickstart.ipynb`.

Finally, sometimes it is helpful to clear the evaluated content in a
Jupyter notebook. You can do this at any time with `Restart Kernel and
clear all outputs` in the upper menu bar under `Kernel`.

#### Obtain lesson materials

The lesson materials will be already available in your supercomputer's
Desktop directory. If for whatever reason these are corrupted, re-obtain
the materials by either:
<span style='color:red'>
a. Copying the source lesson material from
`/packages/public/sol-tutorials/python-comp-math`
or
b. Copying the source lesson materials from the internet.
</span>

------------------------------------------------------------------------

[zipfile2]: ../episodes/files/code/python-novice-inflammation-code.zip
[sslvpn]: https://sslvpn.asu.edu
[sslvpn-mac-fix]: https://asurc.atlassian.net/wiki/spaces/RC/pages/1905262641/Troubleshooting+The+CiscoVPN#MacOS-Systems
[web-portal]: https://sol.asu.edu



