#!/bin/bash
set -e  # Exit on error

echo "Downloading and extracting ELGG database..."
wget --progress=bar:force -O - --no-check-certificate http://datasets.epfl.ch/cloudsuite/ELGG_DB.tar.gz | tar -zxvf -

# workaround for overlayfs
find /var/lib/mysql -type f -exec touch {} \;

# Correctly set root password
root_password="root"

# Determine MySQL/MariaDB service name
MY_SQL=$(find /etc/init.d -name "*mariadb*")
if [ $MY_SQL ]; then
    MY_SQL="mariadb"
else
    MY_SQL="mysql"
fi

echo "Stopping database service..."
service $MY_SQL stop

echo "Preparing for database restoration..."
rm -rf /var/lib/mysql/*

echo "Restoring database from backup..."
mariabackup --prepare --target-dir=/backup/
mariabackup --move-back --target-dir=/backup/

echo "Setting ownership..."
chown -R mysql:mysql /var/lib/mysql/

echo "Starting database service..."
service $MY_SQL start

# Verify database is running
echo "Verifying database service..."
for i in {1..30}; do
    if mysqladmin ping -h localhost --silent; then
        echo "Database service is running"
        break
    fi
    echo "Waiting for database to start... ($i/30)"
    sleep 2
    if [ $i -eq 30 ]; then
        echo "ERROR: Database failed to start"
        exit 1
    fi
done

# Verify ELGG_DB exists
echo "Verifying ELGG_DB database..."
if mysql -e "USE ELGG_DB" 2>/dev/null; then
    echo "ELGG_DB database exists and is accessible"
else
    echo "ERROR: ELGG_DB database not found or not accessible"
    exit 1
fi

# Keep container running
echo "Database setup complete, entering wait mode"
tail -f /dev/null