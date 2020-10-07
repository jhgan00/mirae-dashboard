# [Django Dashboard - Argon Design](https://appseed.us/admin-dashboards/django-dashboard-argon)

> **Open-Source Admin Dashboard** coded in **Django Framework** by **AppSeed** [Web App Generator](https://appseed.us/app-generator) - features:

- UI Kit: **[Argon Dashboard](https://www.creative-tim.com/product/argon-dashboard?ref=appseed)** (Free version) provided by **Creative-Tim**
- UI-Ready app, SQLite Database, Django Native ORM
- Modular design, clean code-base
- Session-Based Authentication, Forms validation
- Deployment scripts: Docker, Gunicorn / Nginx
- Support via **Github** and [Discord](https://discord.gg/fZC6hup).

> Links

- [Django Dashboard Argon](https://appseed.us/admin-dashboards/django-dashboard-argon) - Product page
- [Django Dashboard Argon - Demo](https://django-dashboard-argon.appseed.us/login/) - LIVE App
- [Django Dashboard Argon - Docs](https://docs.appseed.us/admin-dashboards/django-dashboard-argon/) - Documentation

<br />

## Want more? Go PRO!

PRO versions include **Premium UI Kits**, Lifetime updates and **24/7 LIVE Support** (via [Discord](https://discord.gg/fZC6hup))

| [Django Gradient Dark PRO](https://appseed.us/admin-dashboards/django-dashboard-gradient-dark-pro) | [Django Dashboard Black PRO](https://appseed.us/admin-dashboards/django-dashboard-black-pro) | [Django Dashboard Argon PRO](https://appseed.us/admin-dashboards/django-dashboard-argon-pro) |
| --- | --- | --- |
| [![Django Gradient Dark PRO](https://raw.githubusercontent.com/app-generator/django-dashboard-gradient-dark-pro/master/media/django-dashboard-gradient-dark-pro-screen.png)](https://appseed.us/admin-dashboards/django-dashboard-gradient-dark-pro) | [![Django Dashboard Black PRO](https://raw.githubusercontent.com/app-generator/django-dashboard-black-pro/master/media/django-dashboard-black-pro-screen.png)](https://appseed.us/admin-dashboards/django-dashboard-black-pro) | [![Django Dashboard Argon PRO](https://raw.githubusercontent.com/app-generator/django-dashboard-argon-pro/master/media/django-dashboard-argon-pro-screen.png)](https://appseed.us/admin-dashboards/django-dashboard-argon-pro)

<br />
<br />

![Django Dashboard Argon - Template project provided by AppSeed.](https://raw.githubusercontent.com/app-generator/django-dashboard-argon/master/media/django-dashboard-argon-screen.png)

<br />

## How to use it

```bash
$ # Get the code
$ git clone https://github.com/app-generator/django-dashboard-argon.git
$ cd django-dashboard-argon
$
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ # virtualenv env
$ # .\env\Scripts\activate
$
$ # Install modules
$ # SQLIte version
$ pip3 install -r requirements.txt
$
$ # Create tables
$ python manage.py makemigrations
$ python manage.py migrate
$
$ # Start the application (development mode)
$ python manage.py runserver # default port 8000
$
$ # Start the app - custom port 
$ # python manage.py runserver 0.0.0.0:<your_port>
$
$ # Access the web app in browser: http://127.0.0.1:8000/
```

> Note: To use the app, please access the registration page and **create a new user**. After authentication, the app will unlock the private pages.

<br />

## Deployment

The app is provided with a basic configuration to be executed in [Docker](https://www.docker.com/), [Gunicorn](https://gunicorn.org/), and [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/).

### [Docker](https://www.docker.com/) execution
---

The application can be easily executed in a docker container. The steps:

> Get the code

```bash
$ git clone https://github.com/app-generator/django-dashboard-argon.git
$ cd django-dashboard-argon
```

> Start the app in Docker

```bash
$ sudo docker-compose pull && sudo docker-compose build && sudo docker-compose up -d
```

Visit `http://localhost:5005` in your browser. The app should be up & running.

<br />

### [Gunicorn](https://gunicorn.org/)
---

Gunicorn 'Green Unicorn' is a Python WSGI HTTP Server for UNIX.

> Install using pip

```bash
$ pip install gunicorn
```
> Start the app using gunicorn binary

```bash
$ gunicorn --bind=0.0.0.0:8001 core.wsgi:application
Serving on http://localhost:8001
```

Visit `http://localhost:8001` in your browser. The app should be up & running.


<br />

### [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/)
---

Waitress (Gunicorn equivalent for Windows) is meant to be a production-quality pure-Python WSGI server with very acceptable performance. It has no dependencies except ones that live in the Python standard library.

> Install using pip

```bash
$ pip install waitress
```
> Start the app using [waitress-serve](https://docs.pylonsproject.org/projects/waitress/en/stable/runner.html)

```bash
$ waitress-serve --port=8001 core.wsgi:application
Serving on http://localhost:8001
```

Visit `http://localhost:8001` in your browser. The app should be up & running.

<br />

## Credits & Links

### [Django Admin Dashboards](https://appseed.us/admin-dashboards/django)

Index with UI-ready **admin dashboards** generated by the AppSeed platform in [Django Framework](https://www.djangoproject.com/).
Start fast your next Django project by using functional admin dashboards enhanced with Database, ORM, authentication flow, helpers and deployment scripts.

### What is [Django](https://www.djangoproject.com/)

[Django](https://www.djangoproject.com/) is a Python-based free and open-source web framework, which follows the model-template-view architectural pattern. It is maintained by the Django Software Foundation, an independent organization established as a 501 non-profit. Django's primary goal is to ease the creation of complex, database-driven websites.

### [What is a dashboard](https://en.wikipedia.org/wiki/Dashboard_(business))

A dashboard is a set of pages that are easy to read and offer information to the user in real-time regarding his business. A dashboard usually consists of graphical representations of the current status and trends within an organization. Having a well-designed dashboard will give you the possibility to act and make informed decisions based on the data that your business provides - *definition provided by [Creative-Tim - Free Dashboard Templates](https://www.creative-tim.com/blog/web-design/free-dashboard-templates/?ref=appseed)*.

### [Argon Dashboard](https://www.creative-tim.com/product/argon-dashboard?AFFILIATE=128200)

**[Argon Dashboard](https://www.creative-tim.com/product/argon-dashboard?ref=appseed)** is built with over 100 individual components, giving you the freedom of choosing and combining. All components can take variations in color, that you can easily modify using SASS files and it is open source, and free - provided by **Creative-Tim**.

<br />

---
**[Django Dashboard - Argon Design](https://appseed.us/admin-dashboards/django-dashboard-argon)** - Provided by **AppSeed** [Web App Generator](https://appseed.us/app-generator).
