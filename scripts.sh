#!/bin/bash
sudo apt-get update

sudo apt-get install -y curl vim tmux curl git apache2 libapache2-mod-wsgi postgresql postgresql-contrib python-pip python-dev python-psycopg2 libpq-dev python-numpy python-scipy

# Creating swap memory, because pandas build is very memory-intensive.
# Taken from http://ze.phyr.us/pandas-memory-crash/
sudo dd if=/dev/zero of=/swap1 bs=1M count=5000
sudo mkswap /swap1
sudo chown root:root /swap1
sudo chmod 0600 /swap1
sudo swapon /swap1


sudo pip install -r /var/www/textprediction/requirements.txt


if [ -d /var/www/textprediction/build ]; then
  rm -rf /var/www/textprediction/build
fi



sudo -u postgres psql -c "create database textprediction;"
sudo -u postgres psql -c "alter user postgres with password 'cg';"

echo -e '
local   all             postgres                                md5

# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     peer
# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
# IPv6 local connections:
host    all             all             ::1/128                 md5
# Allow replication connections from localhost, by a user with the
# replication privilege.
#local   replication     postgres                                peer
#host    replication     postgres        127.0.0.1/32            md5
#host    replication     postgres        ::1/128                 md5
' | sudo tee /etc/postgresql/9.3/main/pg_hba.conf

cd /etc/apache2
echo -e '
WSGIScriptAlias / /var/www/textprediction/textpredictions/wsgi.py
WSGIPythonPath /var/www/textprediction
Alias /static/ /var/www/textprediction/staticfiles/
<Directory /var/www/textprediction>
  <Files wsgi.py>
    Allow from all
  </Files>
</Directory>
ServerName localhost
' | sudo tee --append apache2.conf

echo -e '
Listen 80
<IfModule mod_ssl.c>
    Listen 443
</IfModule>
<IfModule mod_gnutls.c>
    Listen 443
</IfModule>
' | sudo tee ports.conf

sudo a2dissite 000-default
sudo /etc/init.d/postgresql reload

echo -e '
manage () {
  # Storing current directory
  _c="$PWD"
  cd /var/www/textprediction
  python manage.py "$@"

  # Going back to initial directory
  cd $_c
}
export manage

# Defining a shell command to reset the database content
reset_db () {
  # Storing current directory
  _c="$PWD"
  # Deleting the database
  PGPASSWORD=cg psql -Upostgres -c "drop database textprediction;"
  PGPASSWORD=cg psql -Upostgres -c "create database textprediction;"
  cd /var/www/textprediction
  python -m textblob.download_corpora
  python manage.py makemigrations
  python manage.py migrate

  # Load the fixture into the database (not currently done). Will look something like
  python manage.py collectstatic --noinput --clear
  python manage.py create_models

  # Going back to initial directory
  cd $_c
}
export reset_db

alias run_dev="manage runserver 0.0.0.0:8000"
alias run_apache="sudo service apache2 restart"
alias printlog="cat /var/log/apache2/error.log"
' | sudo tee --append /home/vagrant/.profile

sh /home/vagrant/.profile

# Deleting migrations
rm -rf /var/www/textprediction/textprediction/migrations

PGPASSWORD=cg psql -Upostgres -c "drop database textprediction;"
PGPASSWORD=cg psql -Upostgres -c "create database textprediction;"
cd /var/www/textprediction
python -m textblob.download_corpora
python manage.py makemigrations
python manage.py migrate

# Load the fixture into the database (not currently done). Will look something like
python manage.py collectstatic --noinput --clear
python manage.py create_models
sudo service apache2 restart
manage runserver 0.0.0.0:8000